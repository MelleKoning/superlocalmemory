# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""An asset's version string must move when the asset does.

``index.html`` referenced 64 static assets. 35 carried a hand-written ``?v=``
and 29 carried nothing, and none was derived from the file it pointed at. During
4.0.10 a change to ``od-memories.js`` shipped with its literal still reading the
previous value, which is the failure mode: the number looks like cache-busting,
nothing checks it, and the next person assumes it is handled.

WHAT THIS IS NOT PROTECTING AGAINST, so nobody widens it on a false premise.
Three mechanisms already stop a stale asset reaching a browser here, and the
first was checked against the running daemon rather than assumed:

  * ``/static/*`` is served ``Cache-Control: no-cache, must-revalidate`` with an
    ETag, so a browser must revalidate rather than serve its cached copy.
  * The unified daemon re-copies the UI tree into the data directory on every
    start, so an upgrade refreshes what it serves.
  * ``index.html`` is itself ``no-cache``.

What these tests protect is the ability to CHANGE that: the revalidation policy
costs a conditional request per asset per page load, 64 of them, and the obvious
fix is a long ``max-age``. That is only safe once versions track content. So the
value here is a policy change becoming possible, plus 64 hand-maintained numbers
becoming zero.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from superlocalmemory.server import asset_versions
from superlocalmemory.server.asset_versions import (
    asset_version,
    render_index,
    rewrite_asset_versions,
)

_VERSIONED = re.compile(r'(?:src|href)="(static/[^"?]+)\?v=([^"]*)"')
_ANY_REF = re.compile(r'(?:src|href)="(static/[^"?#]+)')


def _ui_source() -> Path:
    import superlocalmemory

    return Path(superlocalmemory.__file__).parent / "ui"


@pytest.fixture()
def ui(tmp_path: Path) -> Path:
    """A throwaway copy of the shipped UI tree, safe to edit."""
    dest = tmp_path / "ui"
    shutil.copytree(_ui_source(), dest)
    asset_versions._CACHE.clear()
    return dest


def _versions(html: str) -> dict[str, str]:
    return dict(_VERSIONED.findall(html))


class TestTheVersionComesFromTheFile:
    def test_editing_a_stylesheet_moves_its_version(self, ui: Path) -> None:
        """The whole point, stated as the thing that used to be false."""
        target = "static/css/design-system.css"
        on_disk = ui / target[len("static/"):]
        assert on_disk.exists(), "fixture asset moved; pick another"

        before = _versions(render_index(ui / "index.html", ui))
        on_disk.write_text(on_disk.read_text(encoding="utf-8") + "\n/* an edit */\n", encoding="utf-8")
        after = _versions(render_index(ui / "index.html", ui))

        assert target in before and target in after
        assert before[target] != after[target], (
            "the stylesheet changed and its served version did not, which is "
            "the exact defect this exists to prevent"
        )

    def test_only_the_edited_asset_moves(self, ui: Path) -> None:
        """A version that changes when unrelated files change is a false alarm.

        It would also defeat the caching policy this unblocks: one edit
        invalidating 64 URLs is barely better than no versioning.
        """
        target = "static/css/design-system.css"
        before = _versions(render_index(ui / "index.html", ui))
        path = ui / target[len("static/"):]
        path.write_text(path.read_text(encoding="utf-8") + "\n/* an edit */\n", encoding="utf-8")
        after = _versions(render_index(ui / "index.html", ui))

        moved = {k for k in before if before[k] != after.get(k)}
        assert moved == {target}, f"unrelated versions also moved: {moved - {target}}"

    def test_a_touch_without_a_content_change_is_not_a_new_version(
        self, ui: Path,
    ) -> None:
        """Content, not timestamp.

        mtime is used to decide whether to re-read, which is a cost question.
        The version itself has to come from bytes, or reinstalling the same
        release would invalidate every asset for every user.
        """
        target = "static/css/design-system.css"
        path = ui / target[len("static/"):]
        before = _versions(render_index(ui / "index.html", ui))

        content = path.read_bytes()
        path.write_bytes(content)  # new mtime, identical bytes
        asset_versions._CACHE.clear()

        after = _versions(render_index(ui / "index.html", ui))
        assert after[target] == before[target]

    def test_two_files_with_the_same_bytes_get_the_same_version(
        self, ui: Path,
    ) -> None:
        source = ui / "css" / "design-system.css"
        twin = ui / "css" / "_twin-probe.css"
        twin.write_bytes(source.read_bytes())
        assert asset_version(source) == asset_version(twin)

    def test_every_reference_gets_one(self, ui: Path) -> None:
        """29 of the 64 references had no cache-buster at all.

        Those were the ones a policy change would break first, so leaving them
        bare would make this a partial fix that reads as a complete one.
        """
        html = render_index(ui / "index.html", ui)
        assert _versions(html), "nothing was rewritten"
        bare = [
            ref for ref in _ANY_REF.findall(html)
            if f'"{ref}?v=' not in html
        ]
        assert bare == [], f"static references with no version: {bare}"


class TestItCannotBreakThePage:
    def test_asset_paths_are_never_altered(self, ui: Path) -> None:
        """A rewriter that mangles a path takes the dashboard down."""
        original = (ui / "index.html").read_text(encoding="utf-8")
        rendered = render_index(ui / "index.html", ui)
        assert set(_ANY_REF.findall(rendered)) == set(_ANY_REF.findall(original))

    def test_an_unresolvable_reference_keeps_its_literal(self, ui: Path) -> None:
        """Degrade to today's behaviour, never to a page that cannot load.

        The literals stay in the HTML precisely so there is something to fall
        back to.
        """
        html = (ui / "index.html").read_text(encoding="utf-8").replace(
            '<script src="static/js/core.js"',
            '<script src="static/js/absent.js?v=deadbeef"',
            1,
        )
        out = rewrite_asset_versions(html, ui)
        assert 'static/js/absent.js?v=deadbeef' in out

    def test_a_missing_asset_yields_no_version_rather_than_raising(
        self, ui: Path,
    ) -> None:
        assert asset_version(ui / "css" / "not-here.css") is None

    def test_the_wrong_ui_root_is_visible_rather_than_silent(
        self, ui: Path, tmp_path: Path,
    ) -> None:
        """``static/`` is a mount point, not a directory on disk.

        Resolving ``static/js/core.js`` under ``ui_root/static/`` finds nothing
        and leaves all 64 literals in place — a no-op that looks like success.
        This pins that the correct root resolves and a wrong one does not, so
        the mistake shows up here rather than in production.
        """
        good = _versions(render_index(ui / "index.html", ui))
        assert len(good) >= 60

        empty = tmp_path / "nothing"
        empty.mkdir()
        asset_versions._CACHE.clear()
        bad = render_index(ui / "index.html", empty)
        # Unchanged from the source HTML: whatever literals it had, no more.
        assert _versions(bad) == _versions((ui / "index.html").read_text(encoding="utf-8"))

    def test_the_version_placeholder_is_still_substituted(self, ui: Path) -> None:
        """It was substituted by the daemon only, so the other two servers
        served ``__SLM_VERSION__`` verbatim and the dashboard's upgrade detector
        never fired there. One shared function now, so they cannot drift again.
        """
        out = render_index(
            ui / "index.html", ui, substitutions={"__SLM_VERSION__": "9.9.9"},
        )
        assert "__SLM_VERSION__" not in out
        assert "9.9.9" in out


class TestItCostsNothingPerRequest:
    def test_a_warm_render_reads_no_asset_files(self, ui: Path) -> None:
        """This runs on every page load of a daemon that stays up for days.

        Hashing 64 files per request would be a real cost; hashing them once and
        checking (size, mtime_ns) afterwards is a stat per asset.
        """
        reads = {"n": 0}
        original = Path.read_bytes

        def counting(self: Path) -> bytes:
            reads["n"] += 1
            return original(self)

        asset_versions._CACHE.clear()
        Path.read_bytes = counting  # type: ignore[method-assign]
        try:
            render_index(ui / "index.html", ui)
            cold = reads["n"]
            reads["n"] = 0
            for _ in range(5):
                render_index(ui / "index.html", ui)
            warm = reads["n"]
        finally:
            Path.read_bytes = original  # type: ignore[method-assign]

        assert cold >= 60, f"only {cold} assets were hashed on a cold render"
        assert warm == 0, f"{warm} asset reads on five warm renders"


class TestEveryServerUsesIt:
    """Three near-identical ``root()`` handlers read this file.

    They had already drifted once — only one substituted the version
    placeholder. A fix applied to one of three is how that happened.
    """

    @pytest.mark.parametrize("module", ["api", "ui", "unified_daemon"])
    def test_the_root_handler_renders_rather_than_reads(self, module: str) -> None:
        import importlib
        import inspect

        mod = importlib.import_module(f"superlocalmemory.server.{module}")
        src = inspect.getsource(mod)
        assert "render_index(" in src, (
            f"server/{module}.py serves index.html without deriving asset "
            "versions, so its pages carry the hand-written literals"
        )
        # And it must degrade rather than fail. This assertion used to forbid
        # `index_path.read_text()` outright, which was wrong: reading the file
        # as written is precisely the right fallback, and forbidding it is what
        # left the route with nowhere to go when the import failed.
        assert 'return index_path.read_text(encoding="utf-8").replace' in src, (
            f"server/{module}.py has no fallback, so a failure in asset "
            "versioning — a cosmetic feature — returns 500 for the whole page"
        )


class TestACosmeticFeatureCannotTakeThePageDown:
    """The dashboard 500'd in production for two minutes because of this.

    ``pip install -e .`` replaced the installed package underneath a running
    daemon. The import of ``asset_versions`` sits inside the route handler, so
    it resolves at REQUEST time — and the main page began answering "Internal
    Server Error" while every other endpoint was healthy.

    A stale version string is a trifle. A blank dashboard is not.
    """

    @pytest.mark.parametrize("module", ["api", "ui", "unified_daemon"])
    def test_the_fallback_is_reachable_from_the_failure(self, module: str) -> None:
        import importlib
        import inspect

        mod = importlib.import_module(f"superlocalmemory.server.{module}")
        src = inspect.getsource(mod)
        # The render must sit inside a try whose except returns the plain file.
        assert "except Exception as exc:  # noqa: BLE001 — serve the page regardless" in src
        assert "asset version rewrite unavailable" in src, (
            "the fallback is silent; an operator cannot tell that versions "
            "stopped being derived"
        )

    def test_the_fallback_produces_a_whole_page(self) -> None:
        """Not just that it returns — that what it returns is usable."""
        import pathlib as _pl

        import superlocalmemory

        ui = _pl.Path(superlocalmemory.__file__).parent / "ui"
        html = (ui / "index.html").read_text(encoding="utf-8").replace(
            "__SLM_VERSION__", superlocalmemory.__version__,
        )
        assert "<html" in html.lower()
        assert "__SLM_VERSION__" not in html
        assert len(_ANY_REF.findall(html)) >= 60, (
            "the fallback page lost its asset references"
        )


class TestTheRoutesAreStillWiredToTheirHandlers:
    """A decorator separated from its function silently rewires a route.

    Adding a module-level helper immediately after ``@router.post("/api/import")``
    left that decorator attached to the helper, so FastAPI registered the route
    with the helper's signature and every upload came back 422. Nothing in the
    module looked wrong; the existing import tests caught it. This checks the
    shape directly, because the failure is invisible on inspection.
    """

    def test_every_route_decorator_precedes_a_route_handler(self) -> None:
        import ast
        import pathlib as _pl

        import superlocalmemory.server.routes as routes_pkg

        offenders: list[str] = []
        for path in sorted(_pl.Path(routes_pkg.__file__).parent.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                routed = any(
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr in {
                        "get", "post", "put", "patch", "delete", "websocket",
                    }
                    for d in node.decorator_list
                )
                # A handler is public and takes request-ish arguments. A private
                # helper carrying a route decorator is the mistake.
                if routed and node.name.startswith("_"):
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
        assert offenders == [], (
            "route decorator attached to a private helper, so the route is "
            f"registered with the wrong signature: {offenders}"
        )
