# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""4.1.18: Step 4d offers the local answer check (Laya) and must fail open —
a bad network, a missing interpreter, or any other problem here is one plain
line, never a crash, and never a blocked setup."""

from __future__ import annotations

import superlocalmemory.cli.setup_wizard as sw
import superlocalmemory.core.laya_runtime as laya_runtime
import superlocalmemory.retrieval.sufficiency as sufficiency_mod


def _cfg():
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    return SLMConfig.for_mode(Mode.A)


def _ready_status(**overrides) -> laya_runtime.LayaRuntimeStatus:
    fields = dict(state="ready", managed=True, python="/laya/venv/bin/python",
                  hf_home="/laya/hf-cache", model_path="/laya/hf-cache/model",
                  model_revision="rev", progress=1.0, step="Ready")
    fields.update(overrides)
    return laya_runtime.LayaRuntimeStatus(**fields)


class TestLayaStepSkipEnv:
    def test_skip_env_var_short_circuits_before_any_platform_check(self, monkeypatch, capsys):
        monkeypatch.setenv("SLM_SKIP_LAYA", "1")

        def _boom():
            raise AssertionError("laya_supported must not be called when skipped")

        monkeypatch.setattr(sufficiency_mod, "laya_supported", _boom)
        config = _cfg()
        sw._run_laya_step(config, interactive=True)

        assert config.retrieval.sufficiency_judge != "laya"
        assert "SLM_SKIP_LAYA" in capsys.readouterr().out


class TestLayaStepPlatform:
    def test_non_apple_silicon_prints_hosted_option_and_never_installs(self, monkeypatch, capsys):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: False)
        called = []
        monkeypatch.setattr(
            laya_runtime, "install", lambda **k: called.append(1) or _ready_status())

        config = _cfg()
        sw._run_laya_step(config, interactive=True)

        assert called == []
        out = capsys.readouterr().out
        assert "Apple Silicon" in out
        assert config.retrieval.sufficiency_judge != "laya"


class TestLayaStepInteractive:
    def test_default_yes_installs_and_persists_config(self, monkeypatch):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "y")
        monkeypatch.setattr(laya_runtime, "install", lambda **k: _ready_status())

        config = _cfg()
        sw._run_laya_step(config, interactive=True)

        assert config.retrieval.sufficiency_judge == "laya"
        assert config.retrieval.sufficiency_python == "/laya/venv/bin/python"
        assert config.retrieval.sufficiency_hf_home == "/laya/hf-cache"
        assert config.retrieval.sufficiency_model == "/laya/hf-cache/model"

        # Persisted: what the next start reads. (Since 4.1.18 the answer
        # check's settings live in their own file, not inside config.json.)
        from superlocalmemory.core.config import SLMConfig

        saved = SLMConfig.load(config.base_dir / "config.json").retrieval
        assert saved.sufficiency_judge == "laya"
        assert saved.sufficiency_model == "/laya/hf-cache/model"

    def test_answering_no_skips_the_install(self, monkeypatch, capsys):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "n")
        called = []
        monkeypatch.setattr(
            laya_runtime, "install", lambda **k: called.append(1) or _ready_status())

        config = _cfg()
        sw._run_laya_step(config, interactive=True)

        assert called == []
        assert config.retrieval.sufficiency_judge != "laya"
        assert "later in the dashboard" in capsys.readouterr().out


class TestLayaStepNonInteractive:
    """M-13 (4.1.18): unattended setup (``--auto``, CI, no terminal) never
    downloads the ~1.1 GB model by itself — only when asked to."""

    def test_skips_the_install_unless_asked(self, monkeypatch, capsys):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.delenv("SLM_INSTALL_LAYA", raising=False)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("non-interactive setup must never prompt")))
        install_calls = []
        monkeypatch.setattr(
            laya_runtime, "install",
            lambda **k: install_calls.append(1) or _ready_status(),
        )

        config = _cfg()
        sw._run_laya_step(config, interactive=False)

        assert install_calls == []
        assert config.retrieval.sufficiency_judge != "laya"
        out = capsys.readouterr().out
        assert "SLM_INSTALL_LAYA=1" in out

    def test_installs_automatically_without_prompting_when_asked(self, monkeypatch):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.setenv("SLM_INSTALL_LAYA", "1")

        def _prompt_must_not_be_called(*a, **k):
            raise AssertionError("non-interactive setup must never prompt")

        monkeypatch.setattr(sw, "_prompt", _prompt_must_not_be_called)
        install_calls = []
        monkeypatch.setattr(
            laya_runtime, "install",
            lambda **k: install_calls.append(1) or _ready_status(),
        )

        config = _cfg()
        sw._run_laya_step(config, interactive=False)

        assert install_calls == [1]
        assert config.retrieval.sufficiency_judge == "laya"


class TestLayaStepFailure:
    def test_install_failure_prints_dashboard_hint_and_leaves_config_untouched(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "y")
        failed = laya_runtime.LayaRuntimeStatus(
            state="failed", error="Not enough free disk space (needs about 1.5 GB)")
        monkeypatch.setattr(laya_runtime, "install", lambda **k: failed)

        config = _cfg()
        sw._run_laya_step(config, interactive=True)

        assert config.retrieval.sufficiency_judge != "laya"
        out = capsys.readouterr().out
        assert "Settings → Answer check" in out
        assert "Not enough free disk space" in out

    def test_unexpected_exception_is_contained_and_leaves_config_untouched(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.setattr(sw, "_prompt", lambda *a, **k: "y")

        def _boom(**k):
            raise RuntimeError("unexpected install crash")

        monkeypatch.setattr(laya_runtime, "install", _boom)

        config = _cfg()
        sw._run_laya_step(config, interactive=True)  # must not raise

        assert config.retrieval.sufficiency_judge != "laya"
        assert "Settings → Answer check" in capsys.readouterr().out


class TestLayaSummaryLine:
    def test_summary_line_hidden_when_skip_env_set(self, monkeypatch):
        monkeypatch.setenv("SLM_SKIP_LAYA", "1")
        assert sw._laya_summary_line_applies() is False

    def test_summary_line_follows_platform_support(self, monkeypatch):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        assert sw._laya_summary_line_applies() is True
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: False)
        assert sw._laya_summary_line_applies() is False

    def test_unattended_summary_never_announces_a_download_it_will_not_do(self, monkeypatch):
        monkeypatch.setattr(sufficiency_mod, "laya_supported", lambda: True)
        monkeypatch.delenv("SLM_INSTALL_LAYA", raising=False)
        assert sw._laya_summary_line_applies(interactive=False) is False
        monkeypatch.setenv("SLM_INSTALL_LAYA", "1")
        assert sw._laya_summary_line_applies(interactive=False) is True


class TestRerunKeepsAnswerCheckChoices:
    def test_rerunning_setup_keeps_every_answer_check_setting(self):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        existing = SLMConfig.for_mode(Mode.B)
        existing.retrieval.sufficiency_judge = "jev"
        existing.retrieval.sufficiency_jev_provider = "openrouter"
        existing.retrieval.sufficiency_jev_consent = True
        existing.retrieval.sufficiency_python = "/laya/venv/bin/python"
        existing.save(mode_change=True)

        rebuilt = sw._build_wizard_config(Mode.A)

        assert rebuilt.retrieval.sufficiency_judge == "jev"
        assert rebuilt.retrieval.sufficiency_jev_provider == "openrouter"
        assert rebuilt.retrieval.sufficiency_jev_consent is True
        assert rebuilt.retrieval.sufficiency_python == "/laya/venv/bin/python"
