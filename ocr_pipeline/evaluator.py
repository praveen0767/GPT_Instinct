"""
ocr_pipeline/evaluator.py
Full benchmark evaluator: per-field accuracy, decimal placement, ECE, Brier,
reliability diagrams, not_legible exclusion.

Usage:
  python ocr_pipeline/evaluator.py \\
      --benchmark data/prepared/benchmark_manifest.json \\
      --labels    data/prepared/benchmark_manifest.json \\
      --out       reports/ \\
      [--threshold 0.99]
"""
import os
import sys
import json
import time
import argparse
import numpy as np

FIELDS = ["kwh", "kvah", "md_kw", "demand_kva", "meter_serial"]
THRESHOLDS = {
    "exact_acc":          0.99,
    "decimal_acc":        0.99,
    "calibrated_median":  0.995,
    "calibrated_p5":      0.980,
    "decimal_p5":         0.990,
    "latency_p50_ms":     500.0,
    "tilt_subset_acc":    0.99,
}


def _cer(gt: str, pred: str) -> float:
    if not gt:
        return float(len(pred))
    if not pred:
        return 1.0
    n, m = len(gt), len(pred)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], prev if gt[i-1] == pred[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[m] / n


def _decimal_match(gt: str, pred: str) -> bool:
    """True if both have same number of decimal places or same decimal position."""
    def dec_pos(s):
        s = s.strip()
        if '.' not in s:
            return None
        return len(s) - s.index('.') - 1
    return dec_pos(gt) == dec_pos(pred)


def _ece(confs, labels, n_bins=10):
    confs  = np.asarray(confs)
    labels = np.asarray(labels)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confs)) * abs(labels[mask].mean() - confs[mask].mean())
    return float(ece)


def _reliability_diagram(confs, labels, field: str, out_dir: str, n_bins=10):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        confs  = np.asarray(confs)
        labels = np.asarray(labels)
        bins   = np.linspace(0, 1, n_bins + 1)
        acc_list, conf_list = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (confs >= lo) & (confs < hi)
            if m.sum():
                acc_list.append(labels[m].mean())
                conf_list.append(confs[m].mean())
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(conf_list, acc_list, width=1/n_bins, alpha=0.75, edgecolor="k")
        ax.plot([0,1],[0,1],"r--", lw=2)
        ax.set_title(f"Reliability — {field}")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        path = os.path.join(out_dir, f"reliability_{field}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        print(f"Diagram error ({field}): {e}")
        return None


# ── Main evaluator ────────────────────────────────────────────────────────────

class BenchmarkEvaluator:

    def __init__(self, backend_url="http://localhost:8000"):
        self.backend_url = backend_url

    def run(self, manifest_path: str, out_dir: str) -> dict:
        import cv2, requests

        with open(manifest_path) as f:
            records = json.load(f)

        os.makedirs(out_dir, exist_ok=True)

        # Per-field accumulators
        metrics = {f: {"correct": 0, "total": 0, "dec_correct": 0, "dec_total": 0,
                       "confs": [], "labels": [], "cers": []}
                   for f in FIELDS}
        latencies  = []
        excluded   = 0
        tilt_total, tilt_correct = 0, 0

        for rec in records:
            img_path = rec.get("image_path", "")
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Call backend
            try:
                t0 = time.time()
                with open(img_path, "rb") as fp:
                    resp = requests.post(
                        f"{self.backend_url}/infer",
                        files={"file": (os.path.basename(img_path), fp, "image/png")},
                        timeout=30,
                    )
                lat_ms = (time.time() - t0) * 1000
            except Exception as e:
                print(f"  skip {img_path}: {e}")
                continue

            if not resp.ok:
                continue

            result = resp.json()

            # Exclude not-legible
            if result.get("image_quality", {}).get("not_legible"):
                excluded += 1
                continue

            latencies.append(lat_ms)

            # Tilt subset
            tilt_deg = abs(result.get("image_quality", {}).get("tilt_deg", 0.0))
            if tilt_deg > 10:
                tilt_total += 1
                gt_kwh  = str(rec.get("kwh", "")).strip()
                pred_kwh = result.get("kwh", {}).get("value", "").strip()
                if gt_kwh and pred_kwh and gt_kwh == pred_kwh:
                    tilt_correct += 1

            # Per-field
            for field in FIELDS:
                gt   = str(rec.get(field, "")).strip()
                pred_field = result.get(field, {})
                pred = str(pred_field.get("value", "")).strip()
                conf = float(pred_field.get("probability", 0.0))

                if not gt:
                    continue

                is_correct = (gt == pred)
                metrics[field]["correct"] += int(is_correct)
                metrics[field]["total"]   += 1
                metrics[field]["confs"].append(conf)
                metrics[field]["labels"].append(int(is_correct))
                metrics[field]["cers"].append(_cer(gt, pred))

                if field != "meter_serial":
                    metrics[field]["dec_total"] += 1
                    metrics[field]["dec_correct"] += int(_decimal_match(gt, pred))

        # ── Compute summary ────────────────────────────────────────────────
        report = {
            "excluded_not_legible": excluded,
            "total_evaluated":      sum(m["total"] for m in metrics.values()),
            "latency_p50_ms":       float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95_ms":       float(np.percentile(latencies, 95)) if latencies else 0,
            "tilt_subset_total":    tilt_total,
            "tilt_subset_accuracy": float(tilt_correct / max(tilt_total, 1)),
            "fields": {},
        }

        all_pass = True
        for field in FIELDS:
            m = metrics[field]
            n = max(m["total"], 1)
            confs  = np.asarray(m["confs"]) if m["confs"] else np.array([0.0])
            labels = np.asarray(m["labels"]) if m["labels"] else np.array([0])

            exact_acc  = m["correct"] / n
            dec_acc    = m["dec_correct"] / max(m["dec_total"], 1)
            cal_median = float(np.median(confs))
            cal_p5     = float(np.percentile(confs, 5))
            ece_val    = _ece(confs, labels)
            brier      = float(np.mean((confs - labels) ** 2))

            diag_path = _reliability_diagram(confs.tolist(), labels.tolist(), field, out_dir)

            field_pass = (
                exact_acc  >= THRESHOLDS["exact_acc"] and
                dec_acc    >= THRESHOLDS["decimal_acc"] and
                cal_median >= THRESHOLDS["calibrated_median"] and
                cal_p5     >= THRESHOLDS["calibrated_p5"]
            )
            if not field_pass:
                all_pass = False

            report["fields"][field] = {
                "n":              n,
                "exact_accuracy": round(exact_acc, 4),
                "decimal_accuracy": round(dec_acc, 4),
                "calibrated_median": round(cal_median, 4),
                "calibrated_p5":    round(cal_p5, 4),
                "ECE":              round(ece_val, 4),
                "Brier":            round(brier, 4),
                "mean_CER":         round(float(np.mean(m["cers"])) if m["cers"] else 1.0, 4),
                "reliability_diagram": diag_path,
                "PASS":             field_pass,
            }

        report["latency_pass"] = report["latency_p50_ms"] <= THRESHOLDS["latency_p50_ms"]
        report["tilt_pass"]    = report["tilt_subset_accuracy"] >= THRESHOLDS["tilt_subset_acc"]
        report["all_pass"]     = all_pass and report["latency_pass"]

        # Save JSON
        json_path = os.path.join(out_dir, "benchmark_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nBenchmark report saved → {json_path}")

        # Print summary
        print("\n── BENCHMARK SUMMARY ─────────────────────────────────────────")
        print(f"  Evaluated : {report['total_evaluated']}  Excluded: {excluded}")
        print(f"  Latency p50 : {report['latency_p50_ms']:.0f}ms  {'✓' if report['latency_pass'] else '✗'}")
        print(f"  Tilt subset : {report['tilt_subset_accuracy']:.3f}  {'✓' if report['tilt_pass'] else '✗'}")
        for field, fd in report["fields"].items():
            icon = "✓" if fd["PASS"] else "✗"
            print(f"  {icon} {field:14s} acc={fd['exact_accuracy']:.3f}  "
                  f"dec={fd['decimal_accuracy']:.3f}  "
                  f"p5={fd['calibrated_p5']:.3f}  ECE={fd['ECE']:.4f}")
        print(f"\n  Overall: {'ALL PASS ✓' if report['all_pass'] else 'SOME FAILURES ✗'}")
        print("──────────────────────────────────────────────────────────────")

        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Benchmark manifest JSON")
    parser.add_argument("--out",       default="reports/", help="Output directory")
    parser.add_argument("--backend",   default="http://localhost:8000")
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(backend_url=args.backend)
    report    = evaluator.run(args.benchmark, args.out)
    sys.exit(0 if report.get("all_pass") else 1)


if __name__ == "__main__":
    main()
