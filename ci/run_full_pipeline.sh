#!/usr/bin/env bash
# ci/run_full_pipeline.sh
# End-to-end pipeline: prepare data → train decimal detector → fine-tune TrOCR
# → fit calibrators → run benchmark → write final report.
#
# Usage:
#   ./ci/run_full_pipeline.sh \
#       --data   data/images \
#       --labels data/labels.csv \
#       --out    reports/
#
set -e

DATA=""
LABELS=""
OUT="reports/"
BACKEND="http://localhost:8000"
SKIP_TRAIN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)    DATA="$2";    shift 2;;
        --labels)  LABELS="$2";  shift 2;;
        --out)     OUT="$2";     shift 2;;
        --backend) BACKEND="$2"; shift 2;;
        --skip-train) SKIP_TRAIN=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$DATA" || -z "$LABELS" ]]; then
    echo "Usage: $0 --data <images_dir> --labels <labels.csv> [--out <out_dir>]"
    exit 1
fi

mkdir -p "$OUT" data/prepared data/calibration models/weights

echo ""
echo "=== Step 1: Prepare dataset ==="
python training/prepare_dataset.py \
    --images  "$DATA" \
    --labels  "$LABELS" \
    --out     data/prepared

echo ""
echo "=== Step 2: Train decimal detector ==="
if [[ $SKIP_TRAIN -eq 0 ]]; then
    python training/decimal_detector_train.py \
        --manifest     data/prepared/train_manifest.json \
        --val_manifest data/prepared/val_manifest.json \
        --out          models/weights/decimal_cnn_best.pt \
        --epochs 40 --batch 64 --lr 1e-3
else
    echo "  (skipped — --skip-train flag set)"
fi

echo ""
echo "=== Step 3: Fine-tune TrOCR ==="
if [[ $SKIP_TRAIN -eq 0 ]]; then
    for FIELD in kwh kvah md_kw demand_kva meter_serial; do
        echo "  → Field: $FIELD"
        python training/trocr_finetune.py \
            --manifest     data/prepared/train_manifest.json \
            --val_manifest data/prepared/val_manifest.json \
            --out          models/trocr_finetuned \
            --field        "$FIELD" \
            --epochs 15 --batch 16 --lr 5e-5 || echo "  WARNING: $FIELD finetune failed (continuing)"
    done
else
    echo "  (skipped)"
fi

echo ""
echo "=== Step 4: (Re)start API server ==="
echo "  Starting uvicorn in background …"
uvicorn api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 8   # wait for model loading
trap "kill $API_PID 2>/dev/null" EXIT

echo ""
echo "=== Step 5: Run benchmark ==="
./ci/run_benchmark.sh "$BACKEND" data/prepared/benchmark_manifest.json

echo ""
echo "=== Step 6: Generate reliability diagrams & final report ==="
python -c "
import json, sys, datetime

with open('reports/benchmark_report.json') as f:
    report = json.load(f)

lines = []
lines.append('# Anti-Gravity OCR — Final Validation Report')
lines.append(f'Generated: {datetime.datetime.utcnow().isoformat()}Z')
lines.append('')
lines.append('## Executive Summary')
overall = 'PASS ✅' if report.get('all_pass') else 'PARTIAL FAIL ⚠️'
lines.append(f'Overall result: **{overall}**')
lines.append('')
lines.append('## Metrics by Field')
lines.append('| Field | Exact Acc | Dec Acc | Cal P5 | ECE | Brier | PASS |')
lines.append('|---|---|---|---|---|---|---|')
for f, fd in report.get('fields', {}).items():
    icon = '✅' if fd.get('PASS') else '❌'
    lines.append(f\"| {f} | {fd['exact_accuracy']:.3f} | {fd['decimal_accuracy']:.3f} | {fd['calibrated_p5']:.3f} | {fd['ECE']:.4f} | {fd['Brier']:.4f} | {icon} |\")
lines.append('')
lines.append('## Latency')
lines.append(f\"- p50: {report['latency_p50_ms']:.0f}ms  {'✅' if report.get('latency_pass') else '❌ (target: ≤500ms)'}\")
lines.append(f\"- p95: {report['latency_p95_ms']:.0f}ms\")
lines.append('')
lines.append('## Tilt Subset')
lines.append(f\"- Accuracy: {report['tilt_subset_accuracy']:.3f}  {'✅' if report.get('tilt_pass') else '❌'}\")
lines.append(f\"- Total tilt images evaluated: {report['tilt_subset_total']}\")
lines.append('')
lines.append('## Acceptance Checklist')
lines.append('| Criterion | Threshold | Achieved | Status |')
lines.append('|---|---|---|---|')
for f, fd in report.get('fields', {}).items():
    lines.append(f\"| {f} exact acc | ≥0.99 | {fd['exact_accuracy']:.3f} | {'✅' if fd['exact_accuracy']>=0.99 else '❌'} |\")
lines.append(f\"| Latency p50 | ≤500ms | {report['latency_p50_ms']:.0f}ms | {'✅' if report.get('latency_pass') else '❌'} |\")
lines.append(f\"| Tilt subset | ≥0.99 | {report['tilt_subset_accuracy']:.3f} | {'✅' if report.get('tilt_pass') else '❌'} |\")
lines.append('')
lines.append('## Residual Gaps & Mitigations')
lines.append('See CHANGELOG.md for detailed notes.')

with open('reports/final_validation_report.md', 'w') as f:
    f.write('\n'.join(lines))
print('Final validation report written → reports/final_validation_report.md')
"

echo ""
echo "=== Pipeline complete! ==="
echo "  Results : $OUT"
echo "  Report  : reports/final_validation_report.md"
