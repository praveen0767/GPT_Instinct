import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from api.app import app

client = TestClient(app)

image_path = "D:\\GPT_instinct\\Screenshot 2026-03-14 014904.png"

print(f"Testing inference on: {image_path}")
with open(image_path, "rb") as f:
    with TestClient(app) as client:
        response = client.post("/infer", files={"file": ("image.png", f, "image/png")})

print(f"Status Code: {response.status_code}")
try:
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Failed to decode JSON. Raw response:")
    print(response.text)
