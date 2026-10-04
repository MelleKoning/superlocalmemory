# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Remote callers never see the SLM computer's paths, home, account or environment."""

from __future__ import annotations

import asyncio
import getpass
import json
from pathlib import Path

import pytest

from superlocalmemory.server import remote_redaction, remote_tool_policy
from superlocalmemory.server.remote_access import PRINCIPAL_SCOPE_KEY, RemotePrincipal

READ_KEY = RemotePrincipal("remote-key", "rk_00000001", "viewer", "read", "default")


@pytest.fixture(autouse=True)
def _fresh_host_strings():
    remote_redaction.clear_cache()
    yield
    remote_redaction.clear_cache()


def _status_payload() -> dict:
    from superlocalmemory.infra.data_root import canonical_data_root

    root = str(canonical_data_root())
    return {
        "success": True, "profile": "default", "version": "4.1.20", "fact_count": 3,
        "base_dir": root, "db_path": f"{root}/memory.db", "pid": 4242,
        "env": {"PATH": "/usr/bin"}, "log_file": f"{Path.home()}/x.log",
        "note_for_admin": f"stored under {root} for {getpass.getuser()}",
        "windows": r"C:\Users\someone\slm\memory.db",
        "results": [{"fact_id": "f1", "content": "deploy notes live in /etc/deploy/notes.md",
                     "source_path": "/Users/someone/repo/a.py"}],
    }


class _Stub:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def __call__(self, scope, receive, send) -> None:
        while (await receive()).get("more_body"):
            pass
        text = json.dumps(self.payload, indent=2)
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [{"type": "text", "text": text}],
            "structuredContent": self.payload, "isError": False}}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _call(payload: dict, principal) -> dict:
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("default")
    app = remote_tool_policy.RemoteToolScopeASGI(_Stub(payload), runtime_for=lambda _s: runtime)
    scope = {"type": "http", "method": "POST", "path": "/mcp/x", "root_path": "/mcp",
             "headers": [], "client": ("127.0.0.1", 1)}
    if principal is not None:
        scope[PRINCIPAL_SCOPE_KEY] = principal
        scope["slm_remote_listener"] = True
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "get_status", "arguments": {}}}).encode()
    queue = [{"type": "http.request", "body": request, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return json.loads(b"".join(m.get("body", b"") for m in sent
                               if m["type"] == "http.response.body"))


def _host_strings() -> list[str]:
    from superlocalmemory.infra.data_root import canonical_data_root

    return [str(canonical_data_root()), str(Path.home()), "/Users/someone", r"C:\Users"]


def test_get_status_for_a_remote_caller_hides_host_details() -> None:
    answer = _call(_status_payload(), READ_KEY)["result"]
    text = answer["content"][0]["text"]
    structured = answer["structuredContent"]
    for blob in (text, json.dumps(structured)):
        for secret in _host_strings():
            assert secret not in blob, secret
        assert "4242" not in blob and "/usr/bin" not in blob
    data = json.loads(text)
    assert data["base_dir"] == remote_redaction.REDACTED
    assert data["db_path"] == remote_redaction.REDACTED
    assert data["pid"] == remote_redaction.REDACTED
    assert data["env"] == remote_redaction.REDACTED
    # Useful, non-host fields survive.
    assert data["version"] == "4.1.20" and data["fact_count"] == 3
    assert data["profile"] == "default"
    # Memory text is the user's data and is returned exactly as written.
    assert data["results"][0]["content"] == "deploy notes live in /etc/deploy/notes.md"


def test_a_local_caller_keeps_the_full_detail() -> None:
    payload = _status_payload()
    answer = _call(payload, principal=None)["result"]
    assert json.loads(answer["content"][0]["text"]) == payload


def test_account_name_is_withheld_from_free_text() -> None:
    user = getpass.getuser()
    if len(user) < 3:
        pytest.skip("account name too short to match safely")
    out = remote_redaction.redact_value({"message": f"owned by {user}"})
    assert user not in out["message"]


def test_redaction_does_not_mutate_its_input() -> None:
    payload = _status_payload()
    before = json.dumps(payload, sort_keys=True)
    remote_redaction.redact_value(payload)
    assert json.dumps(payload, sort_keys=True) == before


def test_non_json_tool_text_is_redacted_too() -> None:
    result = {"content": [{"type": "text", "text": f"db at {Path.home()}/x/memory.db"}]}
    out = remote_redaction.redact_tool_result(result)["content"][0]["text"]
    assert str(Path.home()) not in out and remote_redaction.HOST_PATH in out


# -- diagnostics never carry host paths; memory text is still returned as written -------

_TB = ('Traceback (most recent call last):\n  File "/Users/someone/slm/src/x.py", line 3\n'
       "FileNotFoundError: [Errno 2] No such file: '/Users/someone/.superlocalmemory/a.db'")


def _diagnostic_payload() -> dict:
    return {
        "ok": False,
        "note": f"internal error: {_TB}",
        "error": {"text": _TB, "content": _TB, "code": "E1"},
        "errors": [{"message": _TB}],
        "store_receipt": {"content": "saved from /Users/someone/notes.md", "fact_id": "f9"},
        "answer_check_note": _TB,
        "results": [{"fact_id": "f1", "content": "my traceback was in /Users/someone/x.py"}],
    }


def test_error_note_and_receipt_fields_never_carry_host_paths_remotely() -> None:
    answer = _call(_diagnostic_payload(), READ_KEY)["result"]
    data = json.loads(answer["content"][0]["text"])
    structured = answer["structuredContent"]
    for blob in (data, structured):
        diagnostics = {k: v for k, v in blob.items() if k != "results"}
        assert "/Users/someone" not in json.dumps(diagnostics), diagnostics
    # The memory's own text is the user's data and is returned exactly as stored.
    assert data["results"][0]["content"] == "my traceback was in /Users/someone/x.py"
    assert structured["results"][0]["content"] == "my traceback was in /Users/someone/x.py"


def test_a_failed_tool_result_is_diagnostics_throughout() -> None:
    result = {"isError": True,
              "content": [{"type": "text", "text": json.dumps({"content": _TB})}],
              "structuredContent": {"content": _TB, "text": _TB}}
    out = remote_redaction.redact_tool_result(result)
    assert "/Users/someone" not in json.dumps(out)


def test_a_jsonrpc_error_answer_is_redacted_remotely() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "error": {
        "code": -32603, "message": f"Error executing tool: {_TB}", "data": {"text": _TB}}}
    out = json.loads(remote_tool_policy._redact_call_answer(json.dumps(payload).encode()))
    assert "/Users/someone" not in json.dumps(out)
    assert out["error"]["code"] == -32603


def test_an_unreadable_tool_answer_is_not_forwarded_as_is() -> None:
    out = remote_tool_policy._redact_call_answer(f"not json {_TB}".encode())
    assert b"/Users/someone" not in out
    assert json.loads(out)["error"]["code"] == -32603
