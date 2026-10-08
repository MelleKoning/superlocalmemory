# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The hosted answer check (Jev), over ``httpx.MockTransport`` only.

No test performs real network I/O: every transport is a MockTransport handed
directly to the constructor / ``check_connection`` via the ``transport=``
contract parameter. A ``_Recorder`` captures every request actually put on
the wire so "exactly one request, even on failure" and "the fake key never
appears in the outgoing body" are checked against the real serialized
request, not against the code's intentions.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval.jev_judge import (
    JEV_ENDPOINTS,
    JevSufficiencyJudge,
    check_connection,
)
from superlocalmemory.retrieval.judge_recipe import JudgeDocument, JudgeRecipe

FAKE_KEY = "sk-test-" + "a1B2c3D4" * 4  # 40 printable ASCII chars
FAKE_SECRET_IN_MEMORY = "ghp_" + "x" * 40  # a GitHub-PAT-shaped token


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def _transport(handler):
    recorder = _Recorder()

    def _wrapped(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        return handler(request)

    return httpx.MockTransport(_wrapped), recorder


def _docs(*contents: str) -> list[JudgeDocument]:
    return [JudgeDocument(content=c) for c in contents]


def _noul_response(model: str, values: dict[str, float], **extra) -> httpx.Response:
    answers = {k: {"type": "noul", "noul": v} for k, v in values.items()}
    body = {"model": model, "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}
    body.update(extra)
    return httpx.Response(200, json=body)


@pytest.fixture
def key_store(tmp_path) -> JudgeKeyStore:
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    store.set_key("openrouter", "or-" + FAKE_KEY)
    return store


def _judge(key_store, *, provider="typesafe", transport, top_k=3, recipe=None, timeout_s=2.0):
    kwargs = dict(
        provider=provider, key_store=key_store, timeout_s=timeout_s,
        top_k=top_k, transport=transport,
    )
    if recipe is not None:
        kwargs["recipe"] = recipe
    return JevSufficiencyJudge(**kwargs)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_happy_path_returns_probabilities_in_memory_order(key_store):
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(model, {"m0": 0.9, "m1": 0.2, "m2": 0.6})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport)

    verdict = judge.judge("what is x?", _docs("doc a", "doc b", "doc c"))

    assert verdict is not None
    assert verdict.probabilities == (0.9, 0.2, 0.6)
    assert verdict.backend == "jev"
    assert len(recorder.requests) == 1


def test_ready_requires_a_stored_key(tmp_path):
    empty_store = JudgeKeyStore(slm_home=tmp_path)
    transport, recorder = _transport(lambda r: _noul_response("x", {}))
    judge = _judge(empty_store, transport=transport)
    assert judge.ready is False
    assert judge.judge("q", _docs("a")) is None
    assert len(recorder.requests) == 0  # never even tries


def test_shutdown_makes_ready_false_and_closes_the_client(key_store):
    transport, recorder = _transport(
        lambda r: _noul_response(JEV_ENDPOINTS["typesafe"][1], {"m0": 0.9})
    )
    judge = _judge(key_store, transport=transport)
    judge.judge("q", _docs("a"))  # opens the lazy client
    judge.shutdown()
    assert judge.ready is False
    assert judge.judge("q", _docs("a")) is None
    assert len(recorder.requests) == 1  # the pre-shutdown call only


# ---------------------------------------------------------------------------
# malformed / out-of-range responses -> None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "answers",
    [
        {"m0": {"type": "noul", "noul": 1.5}},             # out of range
        {"m0": {"type": "noul", "noul": -0.1}},             # out of range
        {"m0": {"type": "noul", "noul": float("nan")}},     # not finite
        {"m0": {"type": "noul"}},                           # missing value
        {"m0": {"type": "choice", "noul": 0.5}},            # wrong type
        {"m0": {"type": "noul", "noul": "0.5"}},            # wrong value type (string)
        {"m0": {"type": "noul", "noul": True}},              # bool, not a float
        {},                                                  # no answers at all
        {"wrong_key": {"type": "noul", "noul": 0.5}},       # key mismatch
    ],
)
def test_malformed_responses_return_none(key_store, answers):
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        # Built with a manual json.dumps (allow_nan=True, the stdlib default)
        # rather than httpx's json= helper, which refuses to encode NaN at
        # all — one of the parametrized payloads needs it on the wire so the
        # parser under test (not the test's own response builder) is what
        # rejects it.
        payload = {"model": model, "answers": answers, "usage": {}}
        content = json.dumps(payload, allow_nan=True).encode("utf-8")
        return httpx.Response(200, content=content,
                              headers={"content-type": "application/json"})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


def test_model_mismatch_in_response_returns_none(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response("some-other-model", {"m0": 0.9})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


def test_non_json_body_returns_none(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


# ---------------------------------------------------------------------------
# transport failures -> None, exactly one request
# ---------------------------------------------------------------------------

def test_timeout_returns_none(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


def test_401_returns_none(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


def test_connect_error_returns_none(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(recorder.requests) == 1


def test_a_failed_request_is_never_retried_within_one_judge_call(key_store):
    """A second call is a NEW judge() invocation, not a retry of the first."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": "boom"})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    assert judge.judge("q", _docs("doc a")) is None
    assert len(calls) == 1
    assert judge.judge("q", _docs("doc a")) is None
    assert len(calls) == 2  # one more call, not an internal retry of the first


# ---------------------------------------------------------------------------
# redaction of outgoing text
# ---------------------------------------------------------------------------

def test_fake_key_shaped_string_never_reaches_the_wire(key_store):
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(model, {"m0": 0.5})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    judge.judge("what is x?", _docs(f"the token is {FAKE_SECRET_IN_MEMORY} end"))

    assert len(recorder.requests) == 1
    raw_body = recorder.requests[0].content.decode("utf-8")
    assert FAKE_SECRET_IN_MEMORY not in raw_body
    assert "[redacted]" in raw_body


def test_fake_key_shaped_string_in_the_question_never_reaches_the_wire(key_store):
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(model, {"m0": 0.5})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    judge.judge(f"does {FAKE_SECRET_IN_MEMORY} answer this?", _docs("an ordinary memory"))

    raw_body = recorder.requests[0].content.decode("utf-8")
    assert FAKE_SECRET_IN_MEMORY not in raw_body


def test_a_memory_that_is_only_a_credential_is_refused_entirely(key_store):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _noul_response(JEV_ENDPOINTS["typesafe"][1], {"m0": 0.5, "m1": 0.5})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=2)

    verdict = judge.judge("q", _docs("a fine memory", FAKE_SECRET_IN_MEMORY))

    assert verdict is None
    assert len(calls) == 0  # refused BEFORE sending anything


# ---------------------------------------------------------------------------
# the key never leaks: not in logs, not in exceptions, not in check_connection
# ---------------------------------------------------------------------------

def test_key_never_appears_in_logs_across_every_failure_path(key_store, caplog):
    caplog.set_level(logging.DEBUG)
    scenarios = [
        lambda r: httpx.Response(401, json={"error": "no"}),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)),
        lambda r: httpx.Response(200, content=b"not json"),
    ]
    for handler in scenarios:
        transport, _ = _transport(handler)
        judge = _judge(key_store, transport=transport, top_k=1)
        judge.judge("q", _docs("doc a"))

    assert FAKE_KEY not in caplog.text


def test_key_never_appears_in_a_raised_exception_repr(key_store):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport, _ = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)
    # judge() must swallow this internally (fails open) rather than raise, but
    # guard the contract anyway: if something unexpected did raise, its repr
    # still must not carry the key.
    try:
        judge.judge("q", _docs("doc a"))
    except Exception as exc:  # pragma: no cover — defence in depth
        assert FAKE_KEY not in repr(exc)
        assert FAKE_KEY not in str(exc)


# ---------------------------------------------------------------------------
# provider -> endpoint/model selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["typesafe", "openrouter"])
def test_endpoint_and_model_match_the_pinned_table(key_store, provider):
    expected_url, expected_model = JEV_ENDPOINTS[provider]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(expected_model, {"m0": 0.9})

    transport, recorder = _transport(handler)
    judge = _judge(key_store, provider=provider, transport=transport, top_k=1)

    verdict = judge.judge("q", _docs("doc a"))

    assert verdict is not None
    sent = recorder.requests[0]
    assert str(sent.url) == expected_url
    assert recorder.bodies()[0]["model"] == expected_model


def test_unknown_provider_is_rejected_at_construction(key_store):
    noop_transport = httpx.MockTransport(lambda r: httpx.Response(200))
    with pytest.raises(ValueError):
        _judge(key_store, provider="not-a-provider", transport=noop_transport)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def test_unmeasured_calibration_cannot_abstain(key_store):
    """A recipe never measured for Jev must never let `insufficient` be True."""
    never_measured = JudgeRecipe(
        recipe_id="sufficiency-v999-not-in-calibrations-table",
        question="Does the memory answer the question?",
    )
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        # Deliberately low confidence — would abstain under any real threshold.
        return _noul_response(model, {"m0": 0.01})

    transport, _ = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1, recipe=never_measured)

    verdict = judge.judge("q", _docs("doc a"))

    assert verdict is not None
    assert verdict.insufficient is False
    assert verdict.calibration_status != "measured_small_sample_not_calibrated"


def test_measured_jev_calibration_is_used_by_default(key_store):
    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(model, {"m0": 0.1})  # below the published 0.5 default

    transport, _ = _transport(handler)
    judge = _judge(key_store, transport=transport, top_k=1)

    verdict = judge.judge("q", _docs("doc a"))

    assert verdict is not None
    assert verdict.threshold == 0.5
    assert verdict.insufficient is True
    assert verdict.calibration_status == "measured_synthetic_set_not_calibrated"


# ===========================================================================
# check_connection
# ===========================================================================

def _check_transport(paris_score: float, meeting_score: float, *, model: str):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        keys = list(body["questions"])
        return _noul_response(model, {keys[0]: paris_score, keys[1]: meeting_score})
    return httpx.MockTransport(handler)


def test_check_connection_succeeds_on_the_expected_scores():
    model = JEV_ENDPOINTS["typesafe"][1]
    transport = _check_transport(0.95, 0.05, model=model)
    ok, message = check_connection("typesafe", FAKE_KEY, transport=transport)
    assert ok is True
    assert FAKE_KEY not in message


def test_check_connection_fails_when_scores_are_not_separated():
    model = JEV_ENDPOINTS["typesafe"][1]
    transport = _check_transport(0.4, 0.6, model=model)
    ok, message = check_connection("typesafe", FAKE_KEY, transport=transport)
    assert ok is False
    assert FAKE_KEY not in message


@pytest.mark.parametrize(
    "status,expected_message",
    [
        (401, "The key was not accepted."),
        (403, "The key was not accepted."),
        (402, "The account has no credit left."),
        (429, "Too many requests — try again in a minute."),
        (500, "The service returned an error (HTTP 500)."),
    ],
)
def test_check_connection_maps_http_status_to_plain_language(status, expected_message):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "x"})

    ok, message = check_connection("typesafe", FAKE_KEY, transport=httpx.MockTransport(handler))
    assert ok is False
    assert message == expected_message
    assert FAKE_KEY not in message


def test_check_connection_timeout_message():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    ok, message = check_connection("typesafe", FAKE_KEY, transport=httpx.MockTransport(handler))
    assert ok is False
    assert message == "The service did not answer in time."


def test_check_connection_unreachable_message():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    ok, message = check_connection("typesafe", FAKE_KEY, transport=httpx.MockTransport(handler))
    assert ok is False
    assert message == "The service could not be reached."


def test_check_connection_exactly_one_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": "boom"})

    check_connection("typesafe", FAKE_KEY, transport=httpx.MockTransport(handler))
    assert len(calls) == 1


def test_check_connection_unknown_provider_makes_no_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    ok, message = check_connection(
        "not-a-provider", FAKE_KEY, transport=httpx.MockTransport(handler)
    )
    assert ok is False
    assert len(calls) == 0


class TestEachMemoryIsJudgedOnItsOwn:
    """Every memory shares one state; a question that does not name its memory
    and rule out the others lets a good memory lift the score of a bad one."""

    def test_each_question_restates_the_question_and_names_only_its_memory(self) -> None:
        from superlocalmemory.retrieval.jev_judge import _build_request
        from superlocalmemory.retrieval.judge_recipe import ACTIVE_RECIPE

        body = _build_request("jev-1.13.0", "What port does the API use?",
                              ["The API listens on 8080.", "Lunch is at noon."],
                              ACTIVE_RECIPE.question)
        for key in ("m0", "m1"):
            text = body["questions"][key]["instructions"]
            assert text.startswith("Question: What port does the API use?")
            assert f"Judge ONLY memory {key}" in text
            assert f"memory {key} contains the specific information" in text
            other = "m1" if key == "m0" else "m0"
            assert f"memory {other}" not in text
            assert "Read `question` and `memory`" not in text


class TestTheProviderMayNameItsSnapshot:
    """OpenRouter answers a request for ``typesafe/jev-1.13`` with the dated
    snapshot it served, e.g. ``typesafe/jev-1.13-20260917`` (seen on a real call,
    2026-10-02). Rejecting that rejected every OpenRouter answer."""

    @pytest.mark.parametrize("provider", ["openrouter", "typesafe"])
    def test_a_dated_snapshot_of_the_requested_model_is_accepted(self, provider) -> None:
        requested = JEV_ENDPOINTS[provider][1]
        transport = _check_transport(0.95, 0.05, model=f"{requested}-20260917")
        ok, _ = check_connection(provider, FAKE_KEY, transport=transport)
        assert ok is True

    @pytest.mark.parametrize("returned", [
        "typesafe/jev-2.0", "typesafe/jev-1.13x", "typesafe/jev-1.13-latest",
        "other/jev-1.13-20260917", "",
    ])
    def test_any_other_model_is_still_refused(self, returned) -> None:
        transport = _check_transport(0.95, 0.05, model=returned)
        ok, _ = check_connection("openrouter", FAKE_KEY, transport=transport)
        assert ok is False, "a threshold measured on one model must not be read as another's"


def test_the_stated_rule_settles_a_permission_question_under_jev_as_under_laya(key_store):
    """Muse audit 2026-10-08, D2: the permission rule is local text logic, so a question
    one judge answers by it must not abstain under the other. The model's numbers stay."""
    from superlocalmemory.retrieval import answer_question_forms as forms

    model = JEV_ENDPOINTS["typesafe"][1]

    def handler(request: httpx.Request) -> httpx.Response:
        return _noul_response(model, {"m0": 0.08, "m1": 0.06})

    transport, _ = _transport(handler)
    judge = _judge(key_store, transport=transport)
    docs = _docs("Never publish until the owner explicitly approves the release.",
                 "The team uses Rust.")
    verdict = judge.judge("May the agent publish without approval?", docs)
    assert verdict is not None
    assert verdict.probabilities == (0.08, 0.06)
    assert verdict.rule_support == (0,)
    assert forms.settled_by_rule(verdict.calibration_id)
    assert not verdict.insufficient
    plain = judge.judge("what is x?", _docs("doc a", "doc b"))
    assert plain.rule_support == () and not forms.settled_by_rule(plain.calibration_id)
