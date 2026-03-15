#!/usr/bin/env python3
import sys
import json
import csv
import glob
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from api.app import app

def generate_qc_dataset(max_images=20):
    
    # Collect images
    image_paths = []
    # User's recent screenshot
    if Path("D:\\GPT_instinct\\Screenshot 2026-03-14 014904.png").exists():
        image_paths.append("D:\\GPT_instinct\\Screenshot 2026-03-14 014904.png")
    
    # Fill remaining from failed_cases and dataset
    image_paths.extend(glob.glob("failed_cases/*.png"))
    image_paths.extend(glob.glob("dataset/train/*.png")[:max_images])
    
    # Dedup and limit
    image_paths = list(dict.fromkeys(image_paths))[:max_images]
    
    if not image_paths:
        print("No images found to process.")
        return

    csv_path = "qc_review.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Required columns: image_id, field, raw_best, corrected, confidence, flags
        writer.writerow(["image_id", "field", "raw_best", "corrected", "confidence", "flags", "reason"])
        
        with TestClient(app) as client:
            for path in image_paths:
                print(f"Processing: {path}")
                try:
                    with open(path, "rb") as f:
                        response = client.post("/infer", files={"file": (Path(path).name, f, "image/png")})
                    
                    if response.status_code != 200:
                        print(f"  -> Error: {response.status_code}")
                        continue
                    
                    res = response.json()
                    img_name = Path(path).name
                    
                    # Check fields
                    fields_to_check = ["kwh", "kvah", "md_kw", "demand_kva", "meter_serial"]
                    for fld in fields_to_check:
                        f_data = res.get(fld, {})
                        if not f_data or f_data.get("value") == "—":
                            continue
                            
                        conf = f_data.get("probability", 0.0)
                        flags = f_data.get("flags", []) or []
                        reason = f_data.get("reason", "")
                        
                        # Log if confidence < 0.8 OR flags present
                        if conf < 0.8 or flags:
                            raw_ocr = f_data.get("debug", {}).get("raw_ocr", "")
                            val = f_data.get("value", "")
                            writer.writerow([img_name, fld, raw_ocr, val, f"{conf:.3f}", "|".join(flags), reason])
                except Exception as e:
                    print(f"  -> Failed: {e}")
                
    print(f"\nGenerated {csv_path} with low-confidence and flagged extractions for review.")

if __name__ == "__main__":
    generate_qc_dataset()
