"""Dashboard routes for the store check: reading the result needs only read
access; Repair now needs the dashboard's write credential like every write."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import integrity
from superlocalmemory.storage import store_check


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "_store_check_args",
                        lambda request: (tmp_path / "memory.db", tmp_path, "4.1.23"))
    app = FastAPI()
    app.include_router(integrity.router)
    return TestClient(app)


def test_the_summary_carries_the_last_result_and_the_plain_labels(tmp_path, monkeypatch):
    (tmp_path / store_check.STATE_FILE).write_text('{"status": "checked", "to_repair": 3}')
    body = _client(tmp_path, monkeypatch).get("/api/integrity/summary").json()
    assert (body["status"], body["to_repair"], body["running"]) == ("checked", 3, False)
    assert body["labels"] == store_check.FINDINGS


def test_check_starts_once_and_says_when_one_is_running(tmp_path, monkeypatch):
    started = iter([True, False])
    monkeypatch.setattr(store_check, "start_check", lambda *a, **k: next(started))
    client = _client(tmp_path, monkeypatch)
    assert client.post("/api/integrity/check").status_code == 202
    assert client.post("/api/integrity/check").status_code == 409


def test_repair_now_is_refused_without_the_write_credential(tmp_path, monkeypatch):
    def never(*a, **k):
        raise AssertionError("repair started without a write credential")

    monkeypatch.setattr(store_check, "start_repair", never)
    response = _client(tmp_path, monkeypatch).post("/api/integrity/repair-now")
    assert response.status_code in (401, 403)


def test_repair_now_starts_for_a_dashboard_write(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(integrity, "_manage", lambda request: calls.append("manage"))
    monkeypatch.setattr(store_check, "start_repair", lambda *a, **k: calls.append("repair") or True)
    response = _client(tmp_path, monkeypatch).post("/api/integrity/repair-now")
    assert response.status_code == 202 and calls == ["manage", "repair"]
