"""What the dashboard says must be a function of what the daemon reports.

Two ways that stopped being true. The health card read a field that is the
literal string "ok" on every reply the daemon is alive enough to send, so a
daemon that had not finished starting, or whose search was not answering,
displayed as Healthy. And the page's version marker was substituted under a
name the page does not contain, so the script that reloads stale assets after
an upgrade saw an unsubstituted placeholder and gave up — on every upgrade,
since the placeholder was mangled.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import superlocalmemory

_UI = pathlib.Path(superlocalmemory.__file__).parent / "ui"


class TestTheVersionMarkerReachesThePage:
    def _shipped_placeholder(self) -> str:
        html = (_UI / "index.html").read_text(encoding="utf-8")
        found = re.search(r'<meta name="slm-version" content="([^"]+)">', html)
        assert found, "the page carries no version marker at all"
        return found.group(1)

    def test_the_page_and_the_script_agree_on_the_placeholder(self):
        core = (_UI / "js" / "core.js").read_text(encoding="utf-8")
        guard = re.search(r"pageVersion === '([^']+)'", core)
        assert guard, "the script no longer guards on an unsubstituted marker"
        assert guard.group(1) == self._shipped_placeholder()

    @pytest.mark.parametrize("module", ["unified_daemon", "api", "ui"])
    def test_every_serve_path_substitutes_the_placeholder_the_page_contains(
        self, module,
    ):
        """Three paths serve this page and each substituted a different string.
        Two of them were mangled variants that appear nowhere in the page."""
        source = (
            pathlib.Path(superlocalmemory.__file__).parent
            / "server" / f"{module}.py"
        ).read_text(encoding="utf-8")
        used = set(re.findall(r"__SLM_[A-Za-z_]*__", source))
        unknown = used - {self._shipped_placeholder()}
        assert not unknown, (
            f"{module}.py substitutes {sorted(unknown)}, which the page does "
            f"not contain, so the version never reaches it"
        )

    def test_rendering_the_page_puts_a_real_version_in_it(self):
        from superlocalmemory.server.asset_versions import render_index

        placeholder = self._shipped_placeholder()
        rendered = render_index(
            _UI / "index.html", _UI, substitutions={placeholder: "9.9.9-test"},
        )
        served = re.search(
            r'<meta name="slm-version" content="([^"]+)">', rendered,
        ).group(1)
        assert served == "9.9.9-test"


class TestTheHealthCardReportsRealReadiness:
    def _script(self) -> str:
        return (_UI / "js" / "od-health.js").read_text(encoding="utf-8")

    def test_it_does_not_decide_from_a_field_that_never_varies(self):
        """`status` is "ok" on every reply. Deciding from it means the card is
        a constant dressed up as a measurement."""
        script = self._script()
        card = script[script.index("Card 1: Daemon"):script.index("Card 2:")]
        assert "runtime_state" in card, (
            "the daemon card does not read the field that reports readiness"
        )

    def test_every_readiness_the_daemon_can_report_has_a_label(self):
        """The daemon has four readiness values; a card that labels one of them
        and falls through on the rest reports the wrong thing three times."""
        daemon_source = (
            pathlib.Path(superlocalmemory.__file__).parent
            / "server" / "unified_daemon.py"
        ).read_text(encoding="utf-8")
        reported = set(re.findall(r'runtime_state = "([a-z_]+)"', daemon_source))
        assert reported, "the daemon no longer reports a readiness state"

        script = self._script()
        for state in reported:
            assert state in script, (
                f"the daemon can report readiness {state!r} and the dashboard "
                f"has no label for it"
            )

    def test_the_daemon_reports_how_far_behind_the_graph_copy_is(self, monkeypatch):
        """A drain that stops advancing is the failure that does not announce
        itself: every other signal stays green while answers get worse.

        This used to assert that two strings appeared in the daemon's source,
        which passes whatever the function does.
        """
        from superlocalmemory.core import backend_orchestrator
        from superlocalmemory.server.unified_daemon import _projection_health

        class _Orchestrator:
            @staticmethod
            def outbox_health():
                return {"depth": 41, "stalled": 3, "draining": True}

        monkeypatch.setattr(
            backend_orchestrator, "get_orchestrator", lambda: _Orchestrator(),
        )
        report = _projection_health()

        assert report["available"] is True
        assert report["depth"] == 41
        assert report["stalled"] == 3
        assert report["behind"] is True, (
            "an alert should not have to know what a healthy depth looks like "
            "on this store"
        )

    def test_a_drained_queue_is_not_reported_as_behind(self, monkeypatch):
        """The control. If ``behind`` were always true it would say nothing."""
        from superlocalmemory.core import backend_orchestrator
        from superlocalmemory.server.unified_daemon import _projection_health

        class _Idle:
            @staticmethod
            def outbox_health():
                return {"depth": 0, "stalled": 0, "draining": False}

        monkeypatch.setattr(
            backend_orchestrator, "get_orchestrator", lambda: _Idle(),
        )
        report = _projection_health()

        assert report["available"] is True
        assert report["behind"] is False

    def test_that_report_never_raises(self, monkeypatch):
        """A health endpoint that fails because one field could not be computed
        is worse than the missing field.

        The failure has to be *arranged*. This test used to hand the function a
        deliberately broken application object -- but the function reads a
        process singleton and never touched it, so the assertion was really about
        whatever the process happened to have built already. Alone that was
        nothing, so it passed; behind a test that had started an orchestrator it
        failed, and the whole suite went red on a test that never tested this.
        """
        from superlocalmemory.core import backend_orchestrator
        from superlocalmemory.server.unified_daemon import _projection_health

        def _explode():
            raise RuntimeError("nothing works")

        monkeypatch.setattr(backend_orchestrator, "get_orchestrator", _explode)
        result = _projection_health()

        assert isinstance(result, dict)
        assert result.get("available") is False
        assert "nothing works" in result.get("error", "")

    def test_no_orchestrator_at_all_is_reported_as_unavailable(self, monkeypatch):
        """Before the first write there is nothing to be behind on, and that is
        not the same as a healthy empty queue."""
        from superlocalmemory.core import backend_orchestrator
        from superlocalmemory.server.unified_daemon import _projection_health

        monkeypatch.setattr(backend_orchestrator, "get_orchestrator", lambda: None)

        assert _projection_health() == {"available": False}
