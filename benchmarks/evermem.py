"""
Real EverMemBench benchmark for xMemory.

Dataset: EverMind-AI/EverMemBench-Dynamic (HuggingFace), 5 multi-party
workplace topics, 2,400 QA items (1,638 multiple-choice + 762 open-ended)
over ~51k chat messages.

Pipeline: ingest every chat message as one memory unit with
"[Group: G][Speaker: S][time]" context, recall top-k per question with the
current engine (pgvector cosine), answer with an LLM (option text for MC),
judge directly for MC and with an LLM judge for open-ended.

Usage:
    python benchmarks/evermem.py                # full run (5 topics)
    python benchmarks/evermem.py --smoke        # topic 01, 20 QA items
    python benchmarks/evermem.py --topics 01 02 # specific topics
"""

import argparse
import json

from common import DATA_DIR, TOP_K, BenchRunner, fail, save_results, summarize

BASE_URL = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic/resolve/main"
TOPICS = ["01", "02", "03", "04", "05"]


def fetch(topic: str, kind: str) -> dict | list:
    path = DATA_DIR / f"evermem_{kind}_{topic}.json"
    if not path.exists():
        import requests

        fname = f"dialogue.json" if kind == "dialogue" else f"qa_{topic}.json"
        url = f"{BASE_URL}/{topic}/{fname}"
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return json.loads(path.read_text())


def topic_messages(dialogue: list) -> list[dict]:
    messages = []
    for rec in dialogue:
        for group, msgs in (rec.get("dialogues") or {}).items():
            if not msgs:
                continue
            for m in msgs:
                messages.append(
                    {
                        "text": m["dialogue"],
                        "context": (
                            f"[Group: {group}][Speaker: {m['speaker']}][{m.get('time', '')}]"
                        ),
                    }
                )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="EverMemBench benchmark for xMemory")
    parser.add_argument("--smoke", action="store_true", help="topic 01, 20 QA items")
    parser.add_argument("--topics", nargs="*", default=None, help="topic ids, e.g. 01 02")
    args = parser.parse_args()

    topics = args.topics or TOPICS
    if args.smoke:
        topics = topics[:1]

    all_records = []
    for topic in topics:
        dialogue = fetch(topic, "dialogue")
        qa_raw = fetch(topic, "qa")
        qa_items = [
            {
                "question": q["Q"],
                "answer": q["A"],
                "options": q.get("options"),
                "category": "mc" if q.get("options") else "open_ended",
            }
            for q in qa_raw
        ]
        if args.smoke:
            qa_items = qa_items[:20]
        print(f"\n=== Topic {topic}: {len(qa_items)} QA items ===", flush=True)

        runner = BenchRunner(bank_id=f"evermem-{topic}")
        runner.ensure_clean()
        messages = topic_messages(dialogue)
        ingest_s = runner.ingest_messages(messages)
        print(f"Ingested {len(messages)} messages in {ingest_s:.1f}s", flush=True)

        records, qa_s = runner.run_qa(qa_items, concurrency=5)
        s = summarize(records)
        print(f"QA done in {qa_s:.1f}s — accuracy: {s['overall']['accuracy']}%")
        all_records.extend(records)

    summary = summarize(all_records)
    print("\n=== EverMemBench overall ===")
    print(json.dumps(summary, indent=2))
    save_results(
        "evermembench",
        {
            "benchmark": "evermembench-dynamic",
            "mode": "smoke" if args.smoke else "full",
            "topics": topics,
            "summary": summary,
            "records": all_records,
        },
    )


if __name__ == "__main__":
    main()
