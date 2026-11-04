#!/bin/bash
# Real LoCoMo benchmark for xMemory.
# Usage: ./run_locomo.sh [--smoke | --convs N]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# bootstrap: locomo.py importa xmemory del repo root (card 60c0005f — el smoke desde benchmarks/ no lo resuelve)
export PYTHONPATH="$(dirname "$SCRIPT_DIR"):${PYTHONPATH:-}"
exec python3 locomo.py "$@"
