"""Consolidate benchmark results into the canonical reports/evals/ directory.

Takes the latest (or explicitly named) result JSON for each suite
(evermembench, locomo, longmemeval_s), writes a timestamped baseline bundle:

    <workspace>/reports/evals/<run_id>/
        baseline.md          — human-readable summary + stack config
        evermembench.json    — exact copies of the measured/frozen results
        locomo.json
        longmemeval_s.json
        manifest.json        — git SHA, endpoints, models, dataset SHAs, seeds

Usage:
    python benchmarks/consolidate.py [--reports-dir PATH]
        [--evermem PATH] [--locomo PATH] [--longmemeval PATH]
        [--label baseline|rebench]

The frozen FASE 0 reference runs (27-oct-2026, git ca7ab6a9) can be adopted
without re-measurement by pointing the flags at their result files.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import common

WORKSPACE_REPORTS = Path(os_default := "/home/joker/.openclaw/workspace/reports/evals")
WORKSPACE_REPORTS = Path("/home/joker/.openclaw/workspace/reports/evals")

def latest(results_dir: Path, prefix: str) -> Path | None:
    files = sorted(results_dir.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def git_info() -> dict:
    def g(*args: str) -> str:
        try:
            return subprocess.run(
                ["git"] + list(args), capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"sha": g("rev-parse", "HEAD"), "branch": g("branch", "--show-current")}


def md_table(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def suite_md(name: str, data: dict) -> str:
    s = data.get("summary", {})
    ov = s.get("overall", {})
    lines = [f"## {name}", ""]
    lines.append(
        f"- total: {ov.get('total')} · correct: {ov.get('correct')} · "
        f"accuracy: **{ov.get('accuracy')}%** · answer_errors: {ov.get('answer_errors')} · "
        f"judge_errors: {ov.get('judge_errors')}"
    )
    if s.get("by_category"):
        rows = [["category", "total", "correct", "accuracy %"]]
        for k, v in s["by_category"].items():
            rows.append([k, v["total"], v["correct"], v["accuracy"]])
        lines += ["", md_table(rows)]
    if s.get("evidence_recall_at_k"):
        e = s["evidence_recall_at_k"]
        lines.append(f"- evidence recall@{e['k']}: {e['rate']}% ({e['hits']}/{e['total']})")
    if s.get("latency_ms"):
        l = s["latency_ms"]
        lines.append(
            f"- latency: answer p50 {l.get('answer_p50_s')}s · p95 {l.get('answer_p95_s')}s"
        )
    cfg = data.get("config") or {}
    if cfg:
        lines.append(
            f"- config: git {cfg.get('git_sha', '?')} · top_k {cfg.get('top_k', '?')} · "
            f"llm {cfg.get('llm_base_url', '?')} · embed {cfg.get('embed_model', '?')} @ "
            f"{cfg.get('embed_base_url', '?')}"
        )
    if data.get("mode"):
        meta = (
            f"mode={data['mode']}"
            + (f", seed={data['seed']}" if data.get("seed") is not None else "")
            + (f", limit={data['limit']}" if data.get("limit") is not None else "")
        )
        lines.append(f"- run: {meta} · finished {data.get('run', {}).get('finished_at', '?')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evermem", type=Path, default=None)
    parser.add_argument("--locomo", type=Path, default=None)
    parser.add_argument("--longmemeval", type=Path, default=None)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--reports-dir", type=Path, default=WORKSPACE_REPORTS)
    args = parser.parse_args()

    rd = Path(__file__).parent / "results"
    sources = {
        "evermembench": args.evermem or latest(rd, "evermembench"),
        "locomo": args.locomo or latest(rd, "locomo"),
        "longmemeval_s": args.longmemeval or latest(rd, "longmemeval_s"),
    }
    missing = [k for k, v in sources.items() if v is None or not Path(v).exists()]
    if missing:
        raise SystemExit(f"Missing results for: {', '.join(missing)} (pass --<suite> PATH)")

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}-{args.label}"
    out_dir = args.reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {}
    for key, path in sources.items():
        path = Path(path)
        payloads[key] = json.loads(path.read_text())
        shutil.copy2(path, out_dir / f"{key}.json")

    cfg = payloads["longmemeval_s"].get("config", {})
    manifest = {
        "run_id": run_id,
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_info(),
        "sources": {k: str(Path(v)) for k, v in sources.items()},
        "stack": {
            "serving_chat": cfg.get("llm_base_url"),
            "answer_model": cfg.get("answer_model"),
            "embeddings": f"{cfg.get('embed_base_url')} ({cfg.get('embed_model')})",
            "reranker": "not used by engine (pgvector+FTS5 RRF only)",
            "top_k": cfg.get("top_k"),
        },
        "datasets": payloads.get("longmemeval_s", {}).get("dataset", {}),
        "suites": {
            k: {
                "file": f"{k}.json",
                "mode": p.get("mode"),
                "accuracy": p.get("summary", {}).get("overall", {}).get("accuracy"),
            }
            for k, p in payloads.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    header = (
        f"# Eval baseline — {args.label} ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
        f"Suite: xMemory retrieval — engine: Postgres pgvector (dense) + FTS5-style full-text RRF hybrid, no separate reranker\n\n"
    )
    parts = [header]
    for key in ("evermembench", "locomo", "longmemeval_s"):
        parts.append(suite_md(key, payloads[key]))
        parts.append("")
    (out_dir / "baseline.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Consolidated: {out_dir}")


if __name__ == "__main__":
    main()
