import traceback
try:
    from api.app import app, _MODELS, _load_models
    _load_models()
    from fastapi.testclient import TestClient
    client = TestClient(app)
    with open('Screenshot 2026-03-14 113016.png', 'rb') as f:
        res = client.post('/infer', files={'file': ('test.png', f, 'image/png')})
    print(res.status_code)
    print(res.json())
except Exception as e:
    traceback.print_exc()
