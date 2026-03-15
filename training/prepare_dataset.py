"""
training/prepare_dataset.py
Dataset preparation: loads raw meter images + labels CSV,
extracts digit/serial crops, creates train/val/benchmark splits.

Expected labels.csv columns:
  image_path, kwh, kvah, md_kw, demand_kva, meter_serial, [bbox_...columns optional]

Usage:
  python training/prepare_dataset.py \\
      --images data/images \\
      --labels data/labels.csv \\
      --out    data/prepared
"""
import os
import sys
import csv
import json
import random
import shutil
import argparse
import cv2
import numpy as np


def load_labels(labels_csv: str) -> list:
    rows = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def create_splits(rows: list, val_frac=0.10, benchmark_n=100, seed=42):
    """Returns (train_rows, val_rows, benchmark_rows)."""
    random.seed(seed)
    shuffled = rows[:]
    random.shuffle(shuffled)
    bench = shuffled[:benchmark_n]
    rest  = shuffled[benchmark_n:]
    n_val = max(1, int(len(rest) * val_frac))
    val   = rest[:n_val]
    train = rest[n_val:]
    return train, val, bench


def extract_display_crop(image: np.ndarray, row: dict) -> np.ndarray:
    """
    Try to cut the display region from the image.
    If bbox columns are present (x1, y1, x2, y2), use them.
    Otherwise return the upper 55% of the image (heuristic for single-phase meters).
    """
    h, w = image.shape[:2]
    try:
        if all(k in row for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
            x1, y1 = int(float(row["bbox_x1"])), int(float(row["bbox_y1"]))
            x2, y2 = int(float(row["bbox_x2"])), int(float(row["bbox_y2"]))
            crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size > 0:
                return crop
    except Exception:
        pass
    # Heuristic: upper 55%
    return image[:int(h * 0.55), :]


def save_split(rows: list, images_dir: str, out_dir: str, split_name: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    skipped  = 0
    for row in rows:
        img_path = row.get("image_path", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(images_dir, img_path)
        if not os.path.isfile(img_path):
            skipped += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        crop = extract_display_crop(img, row)
        base = os.path.splitext(os.path.basename(img_path))[0]
        crop_path = os.path.join(out_dir, f"{base}_display.png")
        cv2.imwrite(crop_path, crop)
        manifest.append({
            "image_path":   crop_path,
            "orig_path":    img_path,
            "kwh":          row.get("kwh", ""),
            "kvah":         row.get("kvah", ""),
            "md_kw":        row.get("md_kw", ""),
            "demand_kva":   row.get("demand_kva", ""),
            "meter_serial": row.get("meter_serial", ""),
        })
    manifest_path = os.path.join(os.path.dirname(out_dir), f"{split_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {split_name}: {len(manifest)} images saved ({skipped} skipped) → {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory with raw meter images")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--out",    required=True, help="Output directory for prepared data")
    parser.add_argument("--benchmark_n", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading labels from {args.labels} …")
    rows = load_labels(args.labels)
    print(f"  {len(rows)} labelled images found.")

    if len(rows) < args.benchmark_n + 10:
        print(f"WARNING: only {len(rows)} images. Reducing benchmark to {max(1, len(rows)//5)}.")
        args.benchmark_n = max(1, len(rows) // 5)

    train, val, bench = create_splits(rows, benchmark_n=args.benchmark_n)
    print(f"Splits: train={len(train)} val={len(val)} benchmark={len(bench)}")

    os.makedirs(args.out, exist_ok=True)
    save_split(train, args.images, os.path.join(args.out, "train"), "train")
    save_split(val,   args.images, os.path.join(args.out, "val"),   "val")
    save_split(bench, args.images, os.path.join(args.out, "benchmark"), "benchmark")
    print("Done.")


if __name__ == "__main__":
    main()
