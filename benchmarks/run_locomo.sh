#!/bin/bash
# Real LoCoMo benchmark for xMemory.
# Usage: ./run_locomo.sh [--smoke | --convs N]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 locomo.py "$@"
