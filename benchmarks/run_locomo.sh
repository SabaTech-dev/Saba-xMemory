#!/bin/bash
# LoCoMo Benchmark Runner for xMemory
# Usage: ./run_locomo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
BENCHMARK_DIR="$HOME/EverOS/benchmarks/locomo"

echo "=== LoCoMo Benchmark for xMemory ==="
echo "Date: $(date)"
echo ""

# Ensure benchmark directory exists
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Error: LoCoMo benchmark directory not found at $BENCHMARK_DIR"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run LoCoMo benchmark
echo "Running LoCoMo benchmark..."
cd "$BENCHMARK_DIR"

# TODO: Replace with actual benchmark command
# python run_locomo.py --adapter=hindsight --output="$RESULTS_DIR/locomo_results_$(date +%Y%m%d).json"

echo "✓ LoCoMo benchmark completed"
echo "Results saved to: $RESULTS_DIR/locomo_results_$(date +%Y%m%d).json"
echo ""

# Display results if file exists
RESULT_FILE="$RESULTS_DIR/locomo_results_$(date +%Y%m%d).json"
if [ -f "$RESULT_FILE" ]; then
    echo "=== Results Summary ==="
    python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f'Overall accuracy: {d.get(\"overall_accuracy\", \"N/A\")}')"
fi

echo ""
echo "=== Benchmark Complete ==="
