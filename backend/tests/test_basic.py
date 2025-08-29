import json
from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta_ok():
    r = client.get("/api/meta")
    assert r.status_code == 200
    data = r.json()
    assert set(["table", "columns", "schema_map", "extra_numeric_columns", "db_path"]) <= set(data.keys())

