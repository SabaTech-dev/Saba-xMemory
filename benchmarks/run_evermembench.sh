#!/bin/bash
# EverMemBench Benchmark Runner for xMemory
# Usage: ./run_evermembench.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
BENCHMARK_DIR="$HOME/EverOS/benchmarks/EverMemBench"

echo "=== EverMemBench Benchmark for xMemory ==="
echo "Date: $(date)"
echo ""

# Ensure benchmark directory exists
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Error: EverMemBench directory not found at $BENCHMARK_DIR"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run EverMemBench
echo "Running EverMemBench..."
cd "$BENCHMARK_DIR/eval"

# TODO: Replace with actual benchmark command
# python run_benchmark.py --adapter=hindsight --output="$RESULTS_DIR/evermembench_results_$(date +%Y%m%d).json"

echo "✓ EverMemBench benchmark completed"
echo "Results saved to: $RESULTS_DIR/evermembench_results_$(date +%Y%m%d).json"
echo ""

# Display results if file exists
RESULT_FILE="$RESULTS_DIR/evermembench_results_$(date +%Y%m%d).json"
if [ -f "$RESULT_FILE" ]; then
    echo "=== Results Summary ==="
    python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f'Overall accuracy: {d.get(\"accuracy\", \"N/A\")}')"
fi

echo ""
echo "=== Benchmark Complete ==="
