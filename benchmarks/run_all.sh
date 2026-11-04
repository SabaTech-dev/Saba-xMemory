#!/bin/bash
# FASE 0 eval harness — one command, three suites.
#
# Usage:
#   ./run_all.sh --smoke                       # fast sanity check of all 3
#   ./run_all.sh                               # full LoCoMo + EverMem, LME-S subset n=50
#   ./run_all.sh --longmemeval-args "--limit 0"  # full LongMemEval-S (hours)
#   ./run_all.sh --consolidate-only            # bundle existing results into reports/evals/
#
# Suites: EverMemBench-Dynamic (frozen 27-oct-2026 dataset, seedless — QA is
# temperature-0 and retrieval deterministic), LoCoMo (10 convs), LongMemEval-S
# (seeded subset by default; --limit 0 for the full 500).
#
# After each quick-win (QW1-QW3) re-run this script and consolidate with
# --label rebench to compare against the FASE 0 baseline.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONSOLIDATE_ONLY=0
LME_ARGS="--limit 50"
SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --consolidate-only) CONSOLIDATE_ONLY=1; shift ;;
    --longmemeval-args) LME_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"

if [[ $SMOKE -eq 1 ]]; then
  "$PY" evermem.py --smoke
  "$PY" locomo.py --smoke
  "$PY" longmemeval.py --smoke
  LABEL=smoke
elif [[ $CONSOLIDATE_ONLY -eq 0 ]]; then
  "$PY" evermem.py
  "$PY" locomo.py
  # shellcheck disable=SC2086
  "$PY" longmemeval.py $LME_ARGS
  LABEL=rebench
fi

"$PY" consolidate.py --label "$LABEL"
