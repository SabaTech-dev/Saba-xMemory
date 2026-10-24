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

TOP_K = 20
EMBED_DIM = 768

ANSWER_PROMPT = """Here are memories from past conversations.

{memories}

Question: {question}
{options_block}
Reply with just the answer: a short phrase, a date, a name, a number, or the option text."""


def llm_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


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
        return time.time() - start

    def answer_question(self, question: str, options: Optional[list[str]]) -> dict:
        q_emb = self.embedder.embed(question)
        result = self.bank.recall(question, top_k=TOP_K, query_embedding=q_emb)
        memories_block = (
            "\n".join(
                f"- [{m.context}] {m.text}" if m.context else f"- {m.text}" for m in result.memories
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

    def judge(self, question: str, reference: str, predicted: str) -> bool:
        """Direct containment first; LLM judge otherwise."""
        if predicted.strip() == "__ERROR__" or predicted.startswith("__ERROR__"):
            return False
        if predicted.strip().lower() == reference.strip().lower():
            return True
        prompt = (
            "You are a fair evaluator of a question-answering system.\n\n"
            f"Question: {question}\n"
            f"Reference (correct) answer: {reference}\n"
            f"Predicted answer: {predicted}\n\n"
            "Judge whether the predicted answer conveys the same key fact as the "
            "reference. Equivalent wording, paraphrases, or a correct subset of "
            "the reference information count as correct. Only judge it incorrect "
            "when it states something contradictory, or fails to answer the "
            "question.\n\n"
            "First reason in one short sentence, then reply on the last line "
            "with exactly one word: CORRECT or INCORRECT."
        )
        out = chat(JUDGE_MODEL, prompt, self.client).strip()
        return out.upper().endswith("CORRECT") and not out.upper().endswith("INCORRECT")

    def run_qa(
        self,
        qa_items: list[dict[str, Any]],
        concurrency: int = 5,
        progress_every: int = 100,
    ) -> tuple[list[dict], float]:
        """Answer + judge all QA items. Items need question/answer, optional
        options and category. Returns (records, seconds)."""
        start = time.time()
        records: list[dict] = [None] * len(qa_items)  # type: ignore[list-item]

        def work(idx: int, item: dict) -> None:
            options = item.get("options")
            ans = self.answer_question(item["question"], options)
            correct = self.judge(item["question"], str(item["answer"]), ans["answer"])
            records[idx] = {
                "question": item["question"],
                "reference": item["answer"],
                "predicted": ans["answer"],
                "correct": correct,
                "category": item.get("category"),
                "n_retrieved": len(ans["retrieved"]),
            }

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(work, i, it): i for i, it in enumerate(qa_items)}
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  [{self.bank_id}] {done}/{len(qa_items)} QA done", flush=True)
        return records, time.time() - start


def summarize(records: list[dict], category_key: str = "category") -> dict:
    total = [r for r in records if r is not None]
    correct = sum(1 for r in total if r["correct"])
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
