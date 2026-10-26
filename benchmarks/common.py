"""
Shared harness utilities for the xMemory benchmarks (LoCoMo, EverMemBench).

Pipeline: ingest conversations into a MemoryBank (batched embeddings via the
configured embedding provider), then for each QA item: recall top-k memories,
generate an answer with an LLM, and judge it (direct match for multiple
choice, LLM judge for open-ended).

Environment:
    XMEMORY_DB_URL        Postgres connection for the benchmark bank
                          (default: postgresql://xmemory:xmemory@localhost/xmemory_bench,
                          schema variant with vector(768): benchmarks/schema768.sql)
    XM_LLM_BASE_URL       OpenAI-compatible endpoint for answering/judging
                          (default: http://127.0.0.1:4000/v1 — LiteLLM)
    XM_LLM_API_KEY        API key for that endpoint (default: LITELLM master
                          key from ~/services/litellm/.env is NOT read here;
                          set explicitly)
    XM_BENCH_MODEL        chat model (default: mi-qwen38-27b-coldfusion-q4km)
    XM_JUDGE_MODEL        judge model (default: same as XM_BENCH_MODEL)
    XM_EMBED_BASE_URL     embedding endpoint (default: same as XM_LLM_BASE_URL)
    XM_EMBED_MODEL        embedding model (default: mi-nomic-embed-text-v2-moe, 768 dims)
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from xmemory import MemoryBank
from xmemory.embeddings import OpenAIEmbeddingProvider

DATA_DIR = Path(__file__).parent / ".data"
RESULTS_DIR = Path(__file__).parent / "results"

LLM_BASE_URL = os.environ.get("XM_LLM_BASE_URL", "http://127.0.0.1:4000/v1")
LLM_API_KEY = os.environ.get("XM_LLM_API_KEY", "")
EMBED_BASE_URL = os.environ.get("XM_EMBED_BASE_URL", LLM_BASE_URL)
EMBED_API_KEY = os.environ.get("XM_EMBED_API_KEY", LLM_API_KEY)

ANSWER_MODEL = os.environ.get("XM_BENCH_MODEL", "mi-qwen38-27b-coldfusion-q4km")
JUDGE_MODEL = os.environ.get("XM_JUDGE_MODEL", ANSWER_MODEL)
EMBED_MODEL = os.environ.get("XM_EMBED_MODEL", "mi-nomic-embed-text-v2-moe")

TOP_K = int(os.environ.get("XM_TOP_K", "20"))
EMBED_DIM = 768

ANSWER_PROMPT = """Here are memories from past conversations.

{memories}

Question: {question}
{options_block}
Reply with just the answer: a short phrase, a date, a name, a number, or the option text.
The name before each "|" in a memory line is who said it — do not attribute one speaker's words or experiences to anyone else.
Lines marked "(neighbor context)" are the turns immediately before/after a recalled memory; use them to connect facts that span several turns, but base your answer on the actual memories.
For dates and times, convert relative expressions ("last Saturday", "next month") into the absolute date, using the timestamp shown in the memory line.
If the question asks about one specific person, check the memories actually describe that person; if they only describe a different person doing that thing, reply: Not mentioned.
For questions about likelihood, preferences, traits or opinions, answer with the most reasonable inference from the memories — the evidence is often indirect, so reason from what is there; only reply "Not mentioned" when truly nothing relevant exists."""


def llm_client() -> OpenAI:
    # timeout 90s + 1 retry de cliente: sin esto, una request colgada (LiteLLM<->vLLM
    # tras un CUDA OOM) bloquea al worker en read() para siempre y congela as_completed
    # (audit 26-oct: run congelado en QA 100/199, py-spy: workers en httpcore read).
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=90.0, max_retries=1)


def chat(model: str, prompt: str, client: OpenAI, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                return f"__ERROR__: {e}"
            time.sleep(2**attempt)


class BenchRunner:
    """Ingest + QA harness over a MemoryBank for one dataset bank."""

    def __init__(self, bank_id: str, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get(
            "XMEMORY_DB_URL", "postgresql://xmemory:xmemory@localhost/xmemory_bench"
        )
        self.embedder = OpenAIEmbeddingProvider(
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            api_key=EMBED_API_KEY,
            dim=EMBED_DIM,
        )
        self.bank = MemoryBank(
            bank_id=bank_id,
            db_url=self.db_url,
            embedding_dim=EMBED_DIM,
            embedding_provider=self.embedder,
        )
        self.client = llm_client()
        self.bank_id = bank_id
        self.expand = True  # neighbor-context expansion of recalled memories
        # Adjacency index for neighbor-context expansion: dia_id -> (prev, next)
        self.adjacency: dict[str, tuple[Optional[str], Optional[str]]] = {}

    def index_adjacency(self, messages: list[dict[str, Any]]) -> None:
        """Index consecutive turns (by dia_id like 'D1:3') so recalled memories
        can be expanded with their adjacent turns at QA time."""
        by_id = {}
        for m in messages:
            ctx = m.get("context", "")
            if "|" in ctx:
                dia = ctx.rsplit("|", 1)[1].strip()
                if dia:
                    by_id[dia] = m["text"]
        ids = sorted(by_id, key=lambda d: (int(d[1:].split(":")[0]), int(d.split(":")[1])))
        for i, dia in enumerate(ids):
            prev = (
                by_id[ids[i - 1]]
                if i > 0 and ids[i - 1][1:].split(":")[0] == dia[1:].split(":")[0]
                else None
            )
            nxt = (
                by_id[ids[i + 1]]
                if i + 1 < len(ids) and ids[i + 1][1:].split(":")[0] == dia[1:].split(":")[0]
                else None
            )
            self.adjacency[dia] = (prev, nxt)

    def expand_neighbors(self, memories: list) -> list[tuple[str, str, bool]]:
        """Return [(text, context, is_neighbor)] expanding each memory with
        adjacent turns (neighbors carry the memory's own context line)."""
        out: list[tuple[str, str, bool]] = []
        seen = {m.text for m in memories}
        for m in memories:
            out.append((m.text, m.context, False))
            dia = m.context.rsplit("|", 1)[1].strip() if m.context and "|" in m.context else ""
            for nb in self.adjacency.get(dia, (None, None)):
                if nb and nb not in seen:
                    seen.add(nb)
                    out.append((nb, m.context, True))
        return out

    def ensure_clean(self) -> None:
        self.bank.ensure_bank()
        conn = self.bank._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_units WHERE bank_id = %s", (self.bank_id,))

    def ingest_messages(self, messages: list[dict[str, Any]], batch_size: int = 256) -> float:
        """Store one memory unit per message. Each message needs 'text';
        optional 'context'. Returns ingest seconds."""
        start = time.time()
        texts = [m["text"] for m in messages]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(self.embedder.embed_batch(texts[i : i + batch_size]))
        for m, emb in zip(messages, embeddings):
            self.bank.retain(m["text"], context=m.get("context", ""), embedding=emb)
        self.index_adjacency(messages)
        return time.time() - start

    def answer_question(
        self, question: str, options: Optional[list[str]], top_k: Optional[int] = None
    ) -> dict:
        q_emb = self.embedder.embed(question)
        result = self.bank.recall(question, top_k=top_k or TOP_K, query_embedding=q_emb)
        if self.expand:
            expanded = self.expand_neighbors(result.memories)
        else:
            expanded = [(m.text, m.context, False) for m in result.memories]
        memories_block = (
            "\n".join(
                f"- ({'neighbor context' if is_nb else 'memory'}) [{ctx}] {text}"
                if ctx
                else f"- ({'neighbor context' if is_nb else 'memory'}) {text}"
                for text, ctx, is_nb in expanded
            )
            or "(no memories found)"
        )
        options_block = ""
        if options:
            options_block = (
                "Options:\n"
                + "\n".join(f"- {o}" for o in options)
                + "\nReply with the exact option text."
            )
        prompt = ANSWER_PROMPT.format(
            memories=memories_block, question=question, options_block=options_block
        )
        answer = chat(ANSWER_MODEL, prompt, self.client)
        return {
            "answer": answer,
            "retrieved": [m.text for m in result.memories],
            "latency_ms": result.latency_ms,
        }

    def judge(self, question: str, reference: str, predicted: str) -> tuple[bool, bool]:
        """Direct containment first; LLM judge otherwise.

        Returns (correct, judge_call_failed): the second flag distinguishes a
        genuine INCORRECT verdict from a failed judge endpoint call, so the
        harness can retry the latter instead of silently scoring it wrong.
        """
        if predicted.strip() == "__ERROR__" or predicted.startswith("__ERROR__"):
            return False, False  # answer already failed; not a judge failure
        if predicted.strip().lower() == reference.strip().lower():
            return True, False
        prompt = (
            "You are a fair evaluator of a question-answering system.\n\n"
            f"Question: {question}\n"
            f"Reference (correct) answer: {reference}\n"
            f"Predicted answer: {predicted}\n\n"
            "Judge whether the predicted answer conveys the same key fact as the "
            "reference. Equivalent wording, paraphrases, or a correct subset of "
            "the reference information count as correct. Only judge it incorrect "
            "when it states something contradictory, or fails to answer the "
            "question.\n"
            "When both answers are dates or relative time expressions (e.g. "
            "'2022', 'last year', 'the week before 9 June 2023', 'next month'), "
            "resolve each to an absolute date or period first, then compare: "
            "same date/period counts as correct even if expressed differently.\n\n"
            "First reason in one short sentence, then reply on the last line "
            "with exactly one word: CORRECT or INCORRECT."
        )
        out = chat(JUDGE_MODEL, prompt, self.client).strip()
        if out.startswith("__ERROR__"):
            return False, True
        return (
            out.upper().endswith("CORRECT") and not out.upper().endswith("INCORRECT"),
            False,
        )

    def run_qa(
        self,
        qa_items: list[dict[str, Any]],
        concurrency: int = 5,
        progress_every: int = 100,
        top_k: Optional[int] = None,
    ) -> tuple[list[dict], float]:
        """Answer + judge all QA items. Items need question/answer, optional
        options and category. Returns (records, seconds)."""
        start = time.time()
        records: list[dict] = [None] * len(qa_items)  # type: ignore[list-item]

        def work(idx: int, item: dict) -> None:
            try:
                options = item.get("options")
                ans = self.answer_question(item["question"], options, top_k=top_k)
                correct, judge_error = self.judge(
                    item["question"], str(item["answer"]), ans["answer"]
                )
                records[idx] = {
                    "question": item["question"],
                    "reference": item["answer"],
                    "predicted": ans["answer"],
                    "correct": correct,
                    "category": item.get("category"),
                    "n_retrieved": len(ans["retrieved"]),
                    "error": ans["answer"].startswith("__ERROR__"),
                    "judge_error": judge_error,
                }
            except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
                records[idx] = {
                    "question": item.get("question", ""),
                    "reference": item.get("answer", ""),
                    "predicted": f"__ERROR__: {e}",
                    "correct": False,
                    "category": item.get("category"),
                    "n_retrieved": 0,
                    "error": True,
                    "judge_error": False,
                }

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(work, i, it): i for i, it in enumerate(qa_items)}
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  [{self.bank_id}] {done}/{len(qa_items)} QA done", flush=True)

        # Error sweeps: transient endpoint failures (5xx bursts, restarts) must
        # not silently depress the score. Retry errored items serially, with
        # low endpoint pressure, before accepting them as failures.
        for sweep in range(2):
            todo = [
                i
                for i, r in enumerate(records)
                if r is None or r.get("error") or r.get("judge_error")
            ]
            if not todo:
                break
            print(
                f"  [{self.bank_id}] retry sweep {sweep + 1}: {len(todo)} errored item(s)",
                flush=True,
            )
            for i in todo:
                work(i, qa_items[i])
        return records, time.time() - start


def summarize(records: list[dict], category_key: str = "category") -> dict:
    total = [r for r in records if r is not None]
    correct = sum(1 for r in total if r["correct"])
    answer_errors = sum(1 for r in total if str(r.get("predicted", "")).startswith("__ERROR__"))
    judge_errors = sum(1 for r in total if r.get("judge_error"))
    by_cat: dict[str, dict[str, int]] = {}
    for r in total:
        cat = str(r.get(category_key))
        c = by_cat.setdefault(cat, {"total": 0, "correct": 0})
        c["total"] += 1
        c["correct"] += 1 if r["correct"] else 0
    return {
        "overall": {
            "total": len(total),
            "correct": correct,
            "accuracy": round(correct / len(total) * 100, 2) if total else 0.0,
            # Validity audit: endpoint errors counted as incorrect (conservative)
            # but surfaced, so a polluted run is distinguishable from a real score.
            "answer_errors": answer_errors,
            "judge_errors": judge_errors,
        },
        "by_category": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1),
            }
            for k, v in sorted(by_cat.items())
        },
    }


def save_results(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out = RESULTS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Results saved: {out}")
    return out


def fail(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)
