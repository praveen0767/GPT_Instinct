import json
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Ensure backend imports work
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from api.app import app, _MODELS, _load_models

def run_inference(image_path: str) -> dict:
    """Wrapper to call the existing Anti-Gravity pipeline robustly."""
    
    # Check if models are loaded natively, since TestClient bypasses @app.on_event("startup")
    if not _MODELS:
        _load_models()

    with TestClient(app) as client:
        with open(image_path, "rb") as f:
            filename = Path(image_path).name
            response = client.post("/infer", files={"file": (filename, f, "image/png")})
    
    if response.status_code != 200:
        return {"error": response.text}
    
    data = response.json()
    
    # Safely extract floats
    def extract_val(field):
        v = data.get(field, {}).get("value")
        if v == "—" or v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return v

    mapped = {
        "image_id": data.get("image_id", "unknown"),
        "kWh": extract_val("kwh"),
        "kWh_probability": data.get("kwh", {}).get("probability", 0.0) if data.get("kwh") else 0.0,
        "kVAh": extract_val("kvah"),
        "MD_kW": extract_val("md_kw"),
        "Demand_kVA": extract_val("demand_kva"),
        "serial": data.get("meter_serial", {}).get("value") if data.get("meter_serial") else None,
        "serial_probability": data.get("meter_serial", {}).get("probability", 0.0) if data.get("meter_serial") else 0.0,
        "decimal_probability": data.get("md_kw", {}).get("decimal_confidence", 0.0) if data.get("md_kw") else 0.0,
        "qc_flag": data.get("qc_flag", False),
        "flags": data.get("reason_codes", []),
        "raw_response": data
    }
    return mapped
