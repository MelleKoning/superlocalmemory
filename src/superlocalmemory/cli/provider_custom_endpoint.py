# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm provider set custom`` — a CLI path to a keyless custom
OpenAI-compatible endpoint for Mode B or Mode C (#112 part 2).

Before this module, only the dashboard's Settings pane could point Mode B or
Mode C at a self-hosted server (llama.cpp, vLLM, LM Studio, …) without a
key; the CLI's ``slm provider set`` only offered named cloud presets. Split
out of ``cli/setup_wizard.py`` to keep that already-large module from
growing further — this file is a self-contained extension of
``configure_provider()``.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from superlocalmemory.cli.setup_wizard import _prompt, is_interactive


def configure_custom_endpoint_provider(
    config: object,
    *,
    endpoint: str | None,
    api_key: str | None,
    model: str | None,
    target_mode: str | None,
    interactive: bool,
) -> None:
    """Configure a custom OpenAI-compatible endpoint for Mode B or Mode C.

    Accepts a keyless local server (llama.cpp, vLLM, LM Studio, …) the
    same way the dashboard's Settings pane already does. Same endpoint-trust
    rules as the remote reranker (``core.provider_endpoint_trust``): HTTPS is
    required for public hosts and bare hostnames; a numeric private-LAN
    address may use plain HTTP only when ``retrieval.trust_plain_http_lan``
    is True; loopback is always allowed.
    """
    from superlocalmemory.core.config import LLMConfig, SLMConfig
    from superlocalmemory.core.provider_endpoint_trust import validate_provider_endpoint_url
    from superlocalmemory.storage.models import Mode

    updated = config if isinstance(config, SLMConfig) else SLMConfig.load()

    if not endpoint and interactive and is_interactive():
        endpoint = _prompt(
            "  Endpoint URL (e.g. http://192.168.1.50:8041/v1 or "
            "http://localhost:8080/v1): ",
        )
    endpoint = (endpoint or "").strip()
    if not endpoint:
        raise ValueError(
            "provider=custom requires an endpoint URL. Pass --endpoint, e.g.\n"
            "  slm provider set custom --endpoint http://192.168.1.50:8041/v1"
        )

    trust_plain_http_lan = getattr(
        getattr(updated, "retrieval", None), "trust_plain_http_lan", True,
    )
    error = validate_provider_endpoint_url(
        endpoint, trust_plain_http_lan=trust_plain_http_lan, label="the LLM endpoint",
    )
    if error:
        raise ValueError(error)

    if api_key is None and interactive and is_interactive():
        api_key = _prompt(
            "  API key (optional — leave blank for a keyless local server): ",
        )
    api_key = (api_key or "").strip()

    if not model and interactive and is_interactive():
        model = _prompt("  Model name (e.g. llama3.2, local-model): ", "local-model")
    model = (model or "local-model").strip()

    mode_choice = (target_mode or "c").strip().lower()
    if mode_choice not in ("b", "c"):
        raise ValueError(f"--mode must be 'b' or 'c', got {mode_choice!r}")
    mode = Mode.B if mode_choice == "b" else Mode.C

    updated.mode = mode
    updated.llm = LLMConfig(
        # "openai" is the established repo token for "any OpenAI-compatible
        # HTTP endpoint" (matches embedding.provider == "openai"; see
        # remote_reranker_config.py's REMOTE_CROSS_ENCODER_BACKENDS comment).
        provider="openai",
        model=model,
        api_key=api_key,
        api_base=endpoint,
    )
    updated.save(mode_change=True)
    SLMConfig.write_current_mode(mode, updated.base_dir)

    print(f"  Mode: {mode.value.upper()}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Model: {model}")
    print(f"  Key: {'configured' if api_key else 'none (keyless)'}")

    ok, message = test_custom_endpoint_connection(endpoint, api_key, model)
    if ok:
        print(f"  ✓ Connection test: {message}")
    else:
        print(f"  ⚠ Connection test: {message}")


def test_custom_endpoint_connection(
    endpoint: str, api_key: str, model: str,
) -> tuple[bool, str]:
    """Probe a custom OpenAI-compatible endpoint.

    Mirrors ``POST /api/v3/provider/test``'s custom-endpoint branch exactly —
    same acceptance rule (HTTP 200/400/422 all prove the server is reachable
    and speaking the expected protocol; a 400/422 can simply mean the probe's
    placeholder model name is unknown to it) — so the CLI and dashboard never
    disagree about whether a given endpoint is "working".
    """
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed — cannot test the connection"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    base = endpoint.rstrip("/")
    if not base.endswith("chat/completions"):
        base = f"{base}/chat/completions"
    probe = {
        "model": model or "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.post(base, headers=headers, json=probe)
            if resp.status_code in (200, 400, 422):
                return True, f"reachable (HTTP {resp.status_code})"
            resp.raise_for_status()
            return True, "connected"
    except httpx.ConnectError:
        return False, "cannot connect — is the service running?"
    except httpx.HTTPStatusError as exc:
        return False, f"HTTP {exc.response.status_code}: invalid key or endpoint"
    except httpx.TimeoutException:
        return False, "connection timed out after 10 seconds"
    except Exception as exc:  # noqa: BLE001 — a connection test must never crash the CLI
        return False, type(exc).__name__
