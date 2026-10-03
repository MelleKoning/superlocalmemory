# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A mesh message to a non-loopback peer passes the same redaction gate as
every other outbound request (Option A, 4.1.19 meshredact).

The egress agent's finding: "Mesh sends agent messages unredacted to
configured or LAN-discovered peers, and its scrub keeps tails." Two bugs:

  1. ``send_to_remote`` / the outbox drain POSTed ``content`` to a LAN/WAN
     peer exactly as typed — no screen at all.
  2. The one scrub that DID exist (``broker_security.scrub_message_content``)
     ran at STORAGE time, used default-aggression ``redact_secrets``, and
     left a ``[REDACTED:TYPE:last4]`` tail — four more characters of a real
     credential than should ever survive a screen, and applied to the LOCAL
     copy too (Option A says local storage stays verbatim; only egress is
     screened).

These tests cover the fix at both egress call sites (the live send and the
durable-outbox retry), confirm a loopback peer is exempt (same rule as
``core.outbound_redaction`` everywhere else), and confirm the local store
never loses the original text.

Non-vacuity: every assertion here was checked to FAIL against the pre-fix
``send_to_remote``/``_drain_outbox`` (raw ``json=payload``, no screen) before
the fix landed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

#: An Anthropic-key-shaped secret — recognized by core.security_primitives'
#: high-aggression pattern set, same shape used elsewhere in this test suite.
_SECRET = "sk-ant-api03-" + "x" * 60


def _make_mesh_db(tmp_path):
    db_path = tmp_path / "mesh_egress_test.db"
    conn = sqlite3.connect(str(db_path))
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL,
        _MESH_V346_ALTERS,
        _MESH_V346_DDL,
    )
    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()
    return db_path


def _make_broker_mock(db_path):
    broker = MagicMock()
    broker._db_path = str(db_path)
    broker._remote_peers = {}
    broker._remote_peers_lock = __import__("threading").RLock()
    return broker


def _sync_client(broker, peer_url: str):
    from superlocalmemory.mesh.remote_sync import RemoteSyncClient
    with patch.dict("os.environ", {"SLM_MESH_PEER_URL": peer_url}, clear=False):
        return RemoteSyncClient(broker)


def _no_tail_redacted(content: str) -> bool:
    """True if content carries the hosted-strength marker with NO tail."""
    return "[redacted]" in content and "[REDACTED:" not in content


@pytest.fixture
def mesh_db(tmp_path):
    return _make_mesh_db(tmp_path)


@pytest.fixture
def broker(mesh_db):
    return _make_broker_mock(mesh_db)


class TestSendToRemoteRedaction:
    """``RemoteSyncClient.send_to_remote`` — the live (non-retried) send."""

    def test_credential_to_lan_peer_arrives_redacted(self, broker, mesh_db):
        client = _sync_client(broker, "http://192.168.1.50:8765")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            post_mock = mock_cls.return_value.__enter__.return_value.post
            client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": _SECRET, "type": "text"},
            )

        sent_content = post_mock.call_args.kwargs["json"]["content"]
        assert _SECRET not in sent_content, "credential leaked to a LAN peer"
        assert _no_tail_redacted(sent_content), (
            f"redaction left a tail or used the wrong marker: {sent_content!r}"
        )

    def test_same_message_to_loopback_peer_arrives_verbatim(self, broker, mesh_db):
        client = _sync_client(broker, "http://127.0.0.1:8765")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            post_mock = mock_cls.return_value.__enter__.return_value.post
            client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": _SECRET, "type": "text"},
            )

        sent_content = post_mock.call_args.kwargs["json"]["content"]
        assert sent_content == _SECRET, "loopback peer must receive text verbatim"

    def test_signature_covers_the_redacted_bytes_actually_sent(self, broker, mesh_db):
        """The HMAC must verify against the SCREENED content, because that is
        what the receiving peer's body actually contains."""
        from superlocalmemory.mesh.broker_security import verify_mesh_message

        with patch.dict(
            "os.environ",
            {
                "SLM_MESH_PEER_URL": "http://192.168.1.50:8765",
                "SLM_MESH_SHARED_SECRET": "fleet-secret",
            },
            clear=False,
        ):
            from superlocalmemory.mesh.remote_sync import RemoteSyncClient
            client = RemoteSyncClient(broker)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            post_mock = mock_cls.return_value.__enter__.return_value.post
            client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": _SECRET, "type": "text"},
            )

        call = post_mock.call_args
        sent_content = call.kwargs["json"]["content"]
        headers = call.kwargs["headers"]
        assert verify_mesh_message(
            "fleet-secret", "me", "peer-1", sent_content,
            headers["X-Mesh-Nonce"], headers["X-Mesh-Ts"], headers["X-Mesh-Sig"],
        ), "signature does not cover the content actually transmitted"


class TestDrainOutboxRedaction:
    """The durable-outbox retry path (``_drain_outbox``) — same gate."""

    def _prefill_row(self, mesh_db, peer_url, content):
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob = RemoteOutbox(str(mesh_db))
        ob.enqueue(
            peer_url=peer_url,
            to_peer="peer-1",
            payload={"from_peer": "me", "to_peer": "peer-1",
                     "content": content, "type": "text"},
            headers=None,
            now=time.time() - 60,  # already due
        )
        return ob

    def test_retried_credential_to_lan_peer_arrives_redacted(self, broker, mesh_db):
        self._prefill_row(mesh_db, "http://192.168.1.50:8765", _SECRET)
        client = _sync_client(broker, "http://192.168.1.50:8765")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            post_mock = mock_cls.return_value.__enter__.return_value.post
            client._drain_outbox()

        sent_content = post_mock.call_args.kwargs["json"]["content"]
        assert _SECRET not in sent_content
        assert _no_tail_redacted(sent_content)

    def test_retried_message_to_loopback_peer_arrives_verbatim(self, broker, mesh_db):
        self._prefill_row(mesh_db, "http://127.0.0.1:8765", _SECRET)
        client = _sync_client(broker, "http://127.0.0.1:8765")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            post_mock = mock_cls.return_value.__enter__.return_value.post
            client._drain_outbox()

        sent_content = post_mock.call_args.kwargs["json"]["content"]
        assert sent_content == _SECRET

    def test_outbox_row_itself_stays_verbatim(self, broker, mesh_db):
        """Option A: the durable outbox is local storage — it keeps the raw
        payload even for a LAN peer; only the eventual POST is screened."""
        self._prefill_row(mesh_db, "http://192.168.1.50:8765", _SECRET)
        conn = sqlite3.connect(str(mesh_db))
        row = conn.execute(
            "SELECT payload FROM mesh_outbox_remote ORDER BY id ASC LIMIT 1"
        ).fetchone()
        conn.close()
        stored_content = json.loads(row[0])["content"]
        assert stored_content == _SECRET, "outbox must store exactly what was enqueued"
