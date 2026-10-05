# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A mode with no model must say so, not just quietly do less.

Mode A runs with no language model at all — that is the point of it, and it is
why nothing on the store or recall path reaches the network. A summary in Mode A
is assembled from the user's own notes rather than written.

The surfaces already said *what* they did. None said *why*, or what to do about
it. Someone seeing a plainer summary than they expected had no way to learn that
a written one needs a model and which modes have one, and would reasonably
conclude the feature was broken.

These tests pin the explanation: present when there is no model, absent when
there is, and identical wherever it appears.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.config import Mode, SLMConfig
from superlocalmemory.core.mode_capability import llm_capability


class TestModeARefusesByConfigurationNotByAccident:
    """The promise should hold because the mode says so."""

    def test_mode_a_declares_that_it_will_not_use_a_model_for_gists(self) -> None:
        """Three of the four places that build the consolidator pass only a
        database, so the guard held by accident of wiring. A caller that passed
        a model to a Mode A engine would have got a model call on the one mode
        whose whole promise is that nothing does."""
        assert SLMConfig.for_mode(Mode.A).ccq.use_llm_gist is False

    def test_the_other_modes_still_use_one(self) -> None:
        assert SLMConfig.for_mode(Mode.B).ccq.use_llm_gist is True
        assert SLMConfig.for_mode(Mode.C).ccq.use_llm_gist is True

    def test_a_supplied_model_does_not_override_the_mode(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Handing a model to a store that runs without one changes nothing.

        This used to be asserted by reading the source of the check and
        comparing where two strings appeared in it, which passes whatever the
        object does. It now builds the thing and asks it.
        """
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SLM_MODE", "a")

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.encoding.cognitive_consolidator import (
            CognitiveConsolidator,
        )
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.database import DatabaseManager

        config = SLMConfig.load()
        assert config.ccq.use_llm_gist is False, (
            "this test is only meaningful on a store configured to summarise "
            "without a language model"
        )
        db = DatabaseManager(config.db_path)
        db.initialize(schema)

        class _Model:
            def generate(self, *args, **kwargs):
                raise AssertionError("a model was consulted on a mode without one")

        # The three places that build this in production pass only a database.
        assert CognitiveConsolidator(db=db)._config.use_llm_gist is False
        # And supplying one does not change the answer.
        assert CognitiveConsolidator(db=db, llm=_Model())._config.use_llm_gist is False

    def test_an_explicit_setting_still_wins(self, tmp_path, monkeypatch) -> None:
        """The control. Reading the store's settings must not stop a caller
        from stating its own."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SLM_MODE", "a")

        from superlocalmemory.core.config import CCQConfig, SLMConfig
        from superlocalmemory.encoding.cognitive_consolidator import (
            CognitiveConsolidator,
        )
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.database import DatabaseManager

        db = DatabaseManager(SLMConfig.load().db_path)
        db.initialize(schema)
        consolidator = CognitiveConsolidator(
            db=db, config=CCQConfig(use_llm_gist=True),
        )
        assert consolidator._config.use_llm_gist is True


class TestTheExplanationIsPresentWhenItIsNeeded:
    def test_mode_a_carries_a_message_naming_the_next_step(self) -> None:
        capability = llm_capability(SLMConfig.for_mode(Mode.A))

        assert capability["llm_available"] is False
        assert capability["message"], "Mode A must explain itself"
        assert "Mode B" in capability["message"]
        assert "Mode C" in capability["message"]

    def test_a_working_model_carries_no_message(self) -> None:
        """An explanation shown when nothing is wrong is noise, and noise is how
        a real warning stops being read."""
        capability = llm_capability(SLMConfig.for_mode(Mode.B), llm_reachable=True)

        assert capability["llm_available"] is True
        assert capability["message"] == ""

    def test_a_configured_model_that_did_not_answer_says_something_different(
        self,
    ) -> None:
        """"You have no model" and "your model did not respond" are different
        problems with different next steps."""
        unreachable = llm_capability(SLMConfig.for_mode(Mode.B), llm_reachable=False)
        absent = llm_capability(SLMConfig.for_mode(Mode.A))

        assert unreachable["message"]
        assert unreachable["message"] != absent["message"]
        assert "reachable" in unreachable["message"]

    def test_it_never_raises_on_a_config_it_cannot_read(self) -> None:
        """This describes a result; it must not be able to break one."""
        class Opaque:
            @property
            def mode(self):
                raise RuntimeError("no mode here")

        capability = llm_capability(Opaque())

        assert capability["llm_available"] is False
        assert capability["message"]


class TestTheSummaryRouteReturnsIt:
    def test_the_route_attaches_capability_to_its_response(self) -> None:
        """A helper nothing calls explains nothing."""
        import inspect

        from superlocalmemory.server.routes import summaries

        source = inspect.getsource(summaries.get_summary)
        assert "llm_capability" in source
        assert '"capability"' in source

    def test_the_dashboard_renders_the_message(self) -> None:
        """The API is not the surface the user is looking at."""
        from pathlib import Path

        import superlocalmemory

        js = (
            # 4.1.21: the Summaries pane moved to its own file.
            Path(superlocalmemory.__file__).parent / "ui" / "js" / "od-summaries.js"
        ).read_text(encoding="utf-8")
        assert "d.capability" in js or "capability" in js
        assert "cap.message" in js, (
            "the dashboard receives the explanation and does not show it"
        )


class TestAHostedModelWithNoKeyIsNotAvailable:
    """Naming a provider is not the same as having one.

    Switching to Mode C sets a provider and a model name and leaves the key
    empty until someone supplies it — the command-line switch says so at the
    time. This description said the model was available and carried no message,
    so every surface that shows it reported nothing wrong while every
    model-backed feature was about to fall back to assembling from notes.
    """

    def _config_with(self, mode, provider, api_key=""):
        """The real config objects are frozen, so this is the shape the
        description reads, not a mutated copy of one."""
        class _LLM:
            def __init__(self, provider, api_key):
                self.provider = provider
                self.api_key = api_key

        class _Config:
            def __init__(self, mode, provider, api_key):
                self.mode = mode
                self.llm = _LLM(provider, api_key)

        return _Config(mode, provider, api_key)

    def test_a_cloud_provider_without_a_key_is_reported_unavailable(self):
        from superlocalmemory.core.config import Mode

        capability = llm_capability(self._config_with(Mode.C, "openrouter"))
        assert capability["llm_available"] is False
        assert "key" in capability["message"].lower()
        assert "slm provider set" in capability["message"]

    def test_a_cloud_provider_with_a_key_is_reported_available(self):
        from superlocalmemory.core.config import Mode

        capability = llm_capability(
            self._config_with(Mode.C, "openrouter", api_key="a-key-value"),
        )
        assert capability["llm_available"] is True
        assert capability["message"] == ""

    def test_a_local_provider_needs_no_key(self):
        """The control. A model on this machine has nothing to authenticate to,
        and demanding a key there would report a working setup as broken."""
        from superlocalmemory.core.config import Mode

        capability = llm_capability(self._config_with(Mode.B, "ollama"))
        assert capability["llm_available"] is True
        assert capability["message"] == ""

    def test_a_caller_that_already_tried_is_believed(self):
        """A summary route has just tried and knows. Its answer wins."""
        from superlocalmemory.core.config import Mode

        capability = llm_capability(
            self._config_with(Mode.C, "openrouter"), llm_reachable=True,
        )
        assert capability["llm_available"] is True
