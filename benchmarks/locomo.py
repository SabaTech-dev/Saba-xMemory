"""
Real LoCoMo benchmark for xMemory.

Dataset: snap-research/locomo (data/locomo10.json), 10 long conversations,
1,986 QA pairs across 5 categories:
    1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.

Pipeline: ingest every dialogue turn as one memory unit (with session
date/time as context), recall top-k per question with the current engine
(pgvector cosine), answer with an LLM, judge with exact match + LLM judge.

Usage:
    python benchmarks/locomo.py                # full run (10 conversations)
    python benchmarks/locomo.py --smoke        # 1 conversation, 20 QA items
    python benchmarks/locomo.py --convs 3      # first N conversations
"""

import argparse
import json

from common import DATA_DIR, TOP_K, BenchRunner, fail, save_results, summarize

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {
    "1": "multi_hop",
    "2": "temporal",
    "3": "open_domain",
    "4": "single_hop",
    "5": "adversarial",
}


def download_dataset() -> list:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "locomo10.json"
    if not path.exists():
        import requests

        print(f"Downloading LoCoMo dataset from {DATASET_URL} ...")
        resp = requests.get(DATASET_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return json.loads(path.read_text())


def conversation_messages(conv: dict) -> list[dict]:
    messages = []
    conversation = conv["conversation"]
    session_keys = sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        n = sk.split("_")[1]
        date_time = conversation.get(f"session_{n}_date_time", "")
        for turn in conversation[sk]:
            messages.append(
                {
                    "text": turn["text"],
                    "context": f"{turn['speaker']} | {date_time} | {turn.get('dia_id', '')}",
                }
            )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for xMemory")
    parser.add_argument("--smoke", action="store_true", help="1 conversation, 20 QA items")
    parser.add_argument("--convs", type=int, default=0, help="limit to first N conversations")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    data = download_dataset()
    if args.smoke:
        data = data[:1]
    elif args.convs:
        data = data[: args.convs]

    all_records = []
    for conv in data:
        sample_id = conv.get("sample_id", "unknown")
        qa_items = []
        for q in conv["qa"]:
            # Adversarial items have no 'answer': the correct reply is that the
            # information is not present in the conversation.
            answer = q.get("answer", "Not mentioned.")
            qa_items.append(
                {
                    "question": q["question"],
                    "answer": answer,
                    "category": CATEGORIES.get(str(q["category"]), str(q["category"])),
                }
            )
        print(f"\n=== Conversation {sample_id}: {len(qa_items)} QA items ===", flush=True)

        runner = BenchRunner(bank_id=f"locomo-{sample_id}")
        runner.ensure_clean()
        messages = conversation_messages(conv)
        ingest_s = runner.ingest_messages(messages)
        print(f"Ingested {len(messages)} turns in {ingest_s:.1f}s", flush=True)

        records, qa_s = runner.run_qa(qa_items)
        s = summarize(records)
        print(f"QA done in {qa_s:.1f}s — accuracy: {s['overall']['accuracy']}%")
        for rec, item in zip(records, qa_items):
            rec["category"] = item["category"]
        all_records.extend(records)

    summary = summarize(all_records)
    print("\n=== LoCoMo overall ===")
    print(json.dumps(summary, indent=2))
    save_results(
        "locomo",
        {
            "benchmark": "locomo",
            "mode": "smoke" if args.smoke else "full",
            "conversations": len(data),
            "summary": summary,
            "records": all_records,
        },
    )


if __name__ == "__main__":
    main()
