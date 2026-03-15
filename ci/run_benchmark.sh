#!/usr/bin/env bash
# ci/run_benchmark.sh
# Runs the full benchmark against a running API server.
# Exits nonzero if any acceptance threshold is not met.
#
# Usage:
#   ./ci/run_benchmark.sh [--backend http://localhost:8000] [--manifest data/prepared/benchmark_manifest.json]
#
set -e

BACKEND="${1:-http://localhost:8000}"
MANIFEST="${2:-data/prepared/benchmark_manifest.json}"
OUT="reports"

echo "=== Anti-Gravity OCR Benchmark ==="
echo "  Backend : $BACKEND"
echo "  Manifest: $MANIFEST"
echo "  Output  : $OUT"
echo ""

# Check backend is up
if ! curl -sf "$BACKEND/health" > /dev/null 2>&1; then
    echo "ERROR: Backend not responding at $BACKEND"
    echo "  Start with: uvicorn api.app:app --host 0.0.0.0 --port 8000"
    exit 1
fi

# Run evaluator
python ocr_pipeline/evaluator.py \
    --benchmark "$MANIFEST" \
    --out       "$OUT"      \
    --backend   "$BACKEND"

EXITCODE=$?
if [ $EXITCODE -ne 0 ]; then
    echo ""
    echo "BENCHMARK FAILED — one or more acceptance criteria not met."
    echo "See $OUT/benchmark_report.json for details."
    exit 1
fi

echo ""
echo "BENCHMARK PASSED — all acceptance criteria met."
exit 0
