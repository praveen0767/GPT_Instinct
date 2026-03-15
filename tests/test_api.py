import requests
import sys
import json

if len(sys.argv) < 2:
    print("Usage: python test_api.py <image_path>")
    sys.exit(1)

url = "http://localhost:8000/infer"
image_path = sys.argv[1]

try:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.split("\\")[-1], f, "image/jpeg")}
        response = requests.post(url, files=files)

    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error calling API: {e}")
