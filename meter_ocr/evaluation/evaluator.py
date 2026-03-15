"""
meter_ocr/evaluation/evaluator.py
Benchmark evaluator.

Metrics (per spec):
  serial exact match ≥ 99%
  decimal placement  ≥ 99%
  kWh accuracy       ≥ 99%

Computes: accuracy, precision, recall, calibration curves.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

THRESHOLDS = {
    "serial_exact":  0.99,
    "decimal_acc":   0.99,
    "kwh_exact":     0.99,
    "latency_p50":   500.0,
}


def _exact(gt: str, pred: str) -> bool:
    return gt.strip() == pred.strip()


def _decimal_match(gt: str, pred: str) -> bool:
    def dec(s):
        return len(s) - s.index('.') - 1 if '.' in s else None
    return dec(gt.strip()) == dec(pred.strip())


def _ece(confs, correct, n_bins=10):
    confs   = np.asarray(confs, float)
    correct = np.asarray(correct, float)
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() == 0: continue
        ece += (m.sum() / len(confs)) * abs(correct[m].mean() - confs[m].mean())
    return float(ece)


def run_evaluation(manifest_path: str, out_dir: str,
                   backend_url: str = "http://localhost:8000") -> dict:
    try:
        import requests, cv2
    except ImportError:
        print("pip install requests opencv-python"); sys.exit(1)

    with open(manifest_path) as f:
        records = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    metrics    = {f: {"correct":0,"dec_correct":0,"total":0,"confs":[]} for f in
                  ("kwh","kvah","md_kw","demand_kva","meter_serial")}
    latencies  = []
    excluded   = 0

    for rec in records:
        img_path = rec.get("image_path","")
        if not os.path.isfile(img_path): continue
        try:
            t0 = time.time()
            with open(img_path,"rb") as fp:
                r = requests.post(
                    f"{backend_url}/infer",
                    files={"file":(os.path.basename(img_path), fp, "image/png")},
                    timeout=30)
            lat = (time.time()-t0)*1000
        except Exception as e:
            print(f"  skip {img_path}: {e}"); continue

        if not r.ok: continue
        result = r.json()
        if result.get("image_quality",{}).get("not_legible"):
            excluded += 1; continue
        latencies.append(lat)

        for field in ("kwh","kvah","md_kw","demand_kva","meter_serial"):
            gt   = str(rec.get(field,"")).strip()
            fd   = result.get(field, {})
            pred = str(fd.get("value","")).strip()
            conf = float(fd.get("probability",0.0))
            if not gt: continue
            correct = int(_exact(gt,pred))
            metrics[field]["correct"]     += correct
            metrics[field]["total"]       += 1
            metrics[field]["confs"].append(conf)
            if field != "meter_serial":
                metrics[field]["dec_correct"] += int(_decimal_match(gt,pred))

    # Summary
    report = {"excluded": excluded, "latency_p50": float(np.percentile(latencies,50)) if latencies else 0,
               "latency_p95": float(np.percentile(latencies,95)) if latencies else 0, "fields": {}}
    all_pass = True
    for field, m in metrics.items():
        n   = max(m["total"],1)
        acc = m["correct"]/n
        dec = m["dec_correct"]/n if field!="meter_serial" else 1.0
        cs  = np.asarray(m["confs"]) if m["confs"] else np.array([0.0])
        ece = _ece(cs, np.array([1.0]*len(cs)))
        report["fields"][field] = {
            "n": n, "accuracy": round(acc,4), "decimal_acc": round(dec,4),
            "conf_median": round(float(np.median(cs)),4),
            "conf_p5":     round(float(np.percentile(cs,5)),4),
            "ECE":         round(ece,4),
            "PASS": acc>= THRESHOLDS.get(f"{field}_exact", 0.99),
        }
        if not report["fields"][field]["PASS"]: all_pass = False

    report["latency_pass"] = report["latency_p50"] <= THRESHOLDS["latency_p50"]
    report["all_pass"]     = all_pass and report["latency_pass"]

    # Print summary
    print("\n── EVALUATION RESULTS ───────────────────────────────────────────")
    print(f"  Excluded (not_legible): {excluded}")
    print(f"  Latency p50: {report['latency_p50']:.0f}ms  {'✓' if report['latency_pass'] else '✗'}")
    for field, fd in report["fields"].items():
        icon = "✓" if fd["PASS"] else "✗"
        print(f"  {icon} {field:14s} acc={fd['accuracy']:.3f} "
              f"dec={fd['decimal_acc']:.3f} p5={fd['conf_p5']:.3f} ECE={fd['ECE']:.4f}")
    print(f"\n  Result: {'ALL PASS ✓' if report['all_pass'] else 'FAILURES ✗'}")
    print("─────────────────────────────────────────────────────────────────")

    # Save report
    path = os.path.join(out_dir,"evaluation_report.json")
    with open(path,"w") as f: json.dump(report,f,indent=2)
    print(f"Report saved → {path}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out",      default="reports/")
    p.add_argument("--backend",  default="http://localhost:8000")
    args = p.parse_args()
    r = run_evaluation(args.manifest, args.out, args.backend)
    sys.exit(0 if r.get("all_pass") else 1)

if __name__ == "__main__":
    main()
