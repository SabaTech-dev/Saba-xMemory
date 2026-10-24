#!/bin/bash
# Real EverMemBench (Dynamic) benchmark for xMemory.
# Usage: ./run_evermembench.sh [--smoke | --topics 01 02 ...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 evermem.py "$@"
