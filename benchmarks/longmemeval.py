"""LongMemEval-S benchmark for xMemory.

Dataset: xiaowu0162/LongMemEval (HF), LongMemEval-S split — 500 questions,
each with its own haystack of timestamped chat sessions (median 50 sessions).
Question types: single-session-user / single-session-assistant /
single-session-preference / multi-session / temporal-reasoning /
knowledge-update.

Pipeline (mirrors evermem.py/locomo.py): per question, ingest every haystack
turn as one memory unit with "[Session: <sid>][<role>][<date>]" context, recall
top-k with the current engine, answer with an LLM, judge with the shared
semantic judge. Additionally reports evidence-session recall@k (fraction of
items where at least one retrieved memory belongs to an answer session) and
latency percentiles.

The full split ingests ~247k turns (hours). Default is a deterministic seeded
subset (--limit 50 --seed 271028) suitable for the FASE 0 baseline and quick
re-benches after each optimization; pass --limit 0 for the full 500.

Usage:
    python benchmarks/longmemeval.py                # subset n=50 seed=271028
    python benchmarks/longmemeval.py --limit 0      # full 500
    python benchmarks/longmemeval.py --smoke        # n=3 fast sanity check
"""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (
    DATA_DIR,
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    ANSWER_MODEL,
    TOP_K,
    BenchRunner,
    save_results,
    summarize,
)

DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"
DATA_FILE = DATA_DIR / "longmemeval_s.json"
DATASET_SHA256_16 = "08d8dad4be43ee20"  # frozen 2026-10-27, first 16 hex of sha256


def load_items() -> list[dict]:
    if not DATA_FILE.exists():
        import requests

        print(f"Downloading {DATASET_URL} ...")
        resp = requests.get(DATASET_URL, timeout=600)
        resp.raise_for_status()
        DATA_FILE.write_bytes(resp.content)
    return json.loads(DATA_FILE.read_text())


def question_turns(item: dict) -> list[dict]:
    """Flatten the haystack sessions into message dicts with session context."""
    messages = []
    sids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    for i, session in enumerate(item["haystack_sessions"]):
        sid = sids[i] if i < len(sids) else f"session_{i}"
        date = dates[i] if i < len(dates) else ""
        for turn in session:
            messages.append(
                {
                    "text": turn["content"],
                    "context": f"[Session: {sid}][{turn['role']}][{date}]",
                }
            )
    return messages


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(pct / 100 * (len(vals) - 1))))
    return round(vals[idx], 1)



def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S benchmark for xMemory")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="number of questions (deterministic seeded sample); 0 = full 500",
    )
    parser.add_argument("--seed", type=int, default=271028)
    parser.add_argument("--smoke", action="store_true", help="n=3 fast sanity check")
    parser.add_argument("--qa-concurrency", type=int, default=5)
    parser.add_argument("--ingest-concurrency", type=int, default=4)
    args = parser.parse_args()

    items = load_items()
    items = sorted(items, key=lambda it: it["question_id"])  # deterministic order
    limit = 3 if args.smoke else args.limit
    if limit and limit > 0:
        rng = random.Random(args.seed)
        items = rng.sample(items, limit)
    print(f"LongMemEval-S: {len(items)} question(s), seed={args.seed}", flush=True)

    # Ingest phase: parallel across questions (each owns its bank).
    turn_counts: dict[str, int] = {}

    def ingest_one(item: dict) -> float:
        runner = BenchRunner(bank_id=f"lme-s-{item['question_id']}")
        runner.ensure_clean()
        messages = question_turns(item)
        turn_counts[item["question_id"]] = len(messages)
        return runner.ingest_messages(messages)

    ingest_start = time.time()
    with ThreadPoolExecutor(max_workers=args.ingest_concurrency) as ex:
        futures = [ex.submit(ingest_one, it) for it in items]
        for n, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if n % 10 == 0 or n == len(futures):
                print(f"ingest {n}/{len(futures)}", flush=True)
    ingest_s = time.time() - ingest_start
    total_turns = sum(turn_counts.values())
    print(f"Ingested {total_turns} turns in {ingest_s:.1f}s", flush=True)

    # QA phase: one item per question, parallel.
    records: list[dict] = [None] * len(items)  # type: ignore[list-item]
    answer_times: list[float] = []

    def qa_one(idx: int, item: dict) -> None:
        try:
            runner = BenchRunner(bank_id=f"lme-s-{item['question_id']}")
            q_emb = runner.embedder.embed(item["question"])
            start = time.time()
            result = runner.bank.recall(item["question"], top_k=TOP_K, query_embedding=q_emb)
            ans = runner.answer_question(item["question"], None)
            answer_s = time.time() - start
            answer_times.append(answer_s)

            evidence = set(item.get("answer_session_ids") or [])
            hit = False
            retrieved_sessions = set()
            for m in result.memories:
                for sid in evidence:
                    if sid in (m.context or ""):
                        hit = True
                        retrieved_sessions.add(sid)

            correct, judge_error = runner.judge(
                item["question"], str(item["answer"]), ans["answer"]
            )
            records[idx] = {
                "question_id": item["question_id"],
                "question": item["question"],
                "reference": str(item["answer"]),
                "predicted": ans["answer"],
                "correct": correct,
                "category": item["question_type"],
                "question_type": item["question_type"],
                "n_turns": turn_counts.get(item["question_id"], 0),
                "n_retrieved": len(result.memories),
                "evidence_hit": hit,
                "recall_latency_ms": result.latency_ms,
                "answer_s": round(answer_s, 1),
                "ingest_s": None,
                "error": ans["answer"].startswith("__ERROR__"),
                "judge_error": judge_error,
            }
        except Exception as e:  # noqa: BLE001
            records[idx] = {
                "question_id": item.get("question_id", ""),
                "question": item.get("question", ""),
                "reference": str(item.get("answer", "")),
                "predicted": f"__ERROR__: {e}",
                "correct": False,
                "category": item.get("question_type", "unknown"),
                "question_type": item.get("question_type", "unknown"),
                "n_turns": turn_counts.get(item.get("question_id"), 0),
                "n_retrieved": 0,
                "evidence_hit": False,
                "recall_latency_ms": 0,
                "answer_s": 0,
                "ingest_s": None,
                "error": True,
                "judge_error": False,
            }

    qa_start = time.time()
    with ThreadPoolExecutor(max_workers=args.qa_concurrency) as ex:
        futures = {ex.submit(qa_one, i, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"QA {done}/{len(futures)}", flush=True)
    qa_s = time.time() - qa_start

    # Error sweep (serial): transient endpoint failures must not depress score.
    for sweep in range(2):
        todo = [
            i for i, r in enumerate(records) if r is None or r.get("error") or r.get("judge_error")
        ]
        if not todo:
            break
        print(f"retry sweep {sweep + 1}: {len(todo)} errored item(s)", flush=True)
        for i in todo:
            qa_one(i, items[i])

    summary = summarize(records)
    summary["evidence_recall_at_k"] = {
        "k": TOP_K,
        "total": len([r for r in records if r]),
        "hits": sum(1 for r in records if r and r.get("evidence_hit")),
        "rate": round(
            100
            * sum(1 for r in records if r and r.get("evidence_hit"))
            / max(1, len([r for r in records if r])),
            1,
        ),
    }
    summary["latency_ms"] = {
        "answer_p50_s": _percentile(answer_times, 50),
        "answer_p95_s": _percentile(answer_times, 95),
    }

    git_sha = ""
    try:
        import subprocess

        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    print("\n=== LongMemEval-S summary ===")
    print(json.dumps(summary, indent=2))
    save_results(
        "longmemeval_s",
        {
            "benchmark": "longmemeval_s",
            "mode": "smoke" if args.smoke else ("subset" if limit else "full"),
            "limit": limit,
            "seed": args.seed,
            "dataset": {
                "url": DATASET_URL,
                "sha256_16": DATASET_SHA256_16,
                "total_questions": 500,
                "total_turns_full": 246930,
            },
            "config": {
                "git_sha": git_sha,
                "llm_base_url": LLM_BASE_URL,
                "embed_base_url": EMBED_BASE_URL,
                "embed_model": EMBED_MODEL,
                "answer_model": ANSWER_MODEL,
                "judge_model": ANSWER_MODEL,
                "top_k": TOP_K,
                "qa_concurrency": args.qa_concurrency,
                "ingest_concurrency": args.ingest_concurrency,
            },
            "run": {
                "turns_ingested": total_turns,
                "ingest_s": round(ingest_s, 1),
                "qa_s": round(qa_s, 1),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "summary": summary,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
