# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer check: one Save for every setting, Test for the on-device check,
and Cancel for a setup.

Save. The dashboard collects the choice (on this Mac / online with Jev / off),
the Jev provider, a new key, the data-sharing notice and Jev's reordering into
one form and sends them here together. Either all of it is applied — saved to
the settings store and switched on in the running daemon, no restart — or
none of it is, with a plain reason saying what to finish first. Nothing is
half-saved: a key is written only once every check has passed.

One option runs at a time. Choosing Jev means only Jev runs; choosing the
on-device check means nothing is sent to Jev — reordering included, which is
part of Jev. The person's reordering choice is remembered, not reset, and it
runs again when Jev is chosen again.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.core import answer_check_state, judge_keys, judge_selection, laya_runtime
from superlocalmemory.retrieval import sufficiency
from superlocalmemory.server.routes import answer_check as base
from superlocalmemory.server.routes import answer_check_support as support

logger = logging.getLogger("superlocalmemory.server.routes.answer_check")

router = APIRouter(prefix="/api/v3/answer-check", tags=["answer-check"])

#: What changes which check runs, or what it may send. Anything else saved
#: (nothing, today) would not need the running check rebuilt.
_JUDGE_FIELDS = ("sufficiency_judge", "sufficiency_jev_provider", "sufficiency_jev_consent",
                 "sufficiency_jev_rerank", "sufficiency_jev_rerank_consent")


class SaveRequest(BaseModel):
    """Everything on the Answer check form. ``mode`` None keeps the current
    choice (including the never-chosen starting state); ``key`` "" keeps the
    saved key."""

    model_config = ConfigDict(extra="forbid")

    mode: str | None = Field(None, pattern="^(laya|jev|off)$")
    provider: str = Field(..., pattern=base._PROVIDER_PATTERN)
    key: str = Field("", max_length=4096)
    consent: StrictBool
    rerank: StrictBool = False


def _refusal(mode: str, body: SaveRequest, retrieval: Any) -> str:
    """Why this form can't be saved as it is ("" when it can) — always with
    what to do about it."""
    if mode == judge_selection.MODE_LAYA:
        laya = laya_runtime.detect(retrieval)
        if laya.state == laya_runtime.STATE_INSTALLING:
            return "The on-device check is still being set up. Save again when it says Ready."
        if laya.state != laya_runtime.STATE_READY:
            return ("The on-device check isn't ready yet. Finish it under "
                    "'On this Mac' first, then Save.")
    if mode == judge_selection.MODE_JEV:
        if not body.key.strip():
            key = support.key_state(body.provider)
            if key["key_problem"]:
                return key["key_problem"]
            if not key["has_key"]:
                return "Paste your Jev key first, then Save."
        if not body.consent:
            return "Tick the notice about what is sent to Jev, then Save."
    if body.key.strip() and not judge_keys.is_usable_key(body.key.strip()):
        return "That doesn't look like a key. Paste the whole key, then Save."
    return ""


def _apply_form(body: SaveRequest, mode: str):
    def change(values: dict) -> dict:
        rerank = bool(body.rerank and body.consent)
        return {**values, "sufficiency_judge": mode,
                "sufficiency_jev_provider": body.provider,
                "sufficiency_jev_consent": bool(body.consent),
                "sufficiency_jev_rerank": rerank,
                "sufficiency_jev_rerank_consent": rerank}
    return change


def _judge_fields(retrieval: Any) -> tuple:
    snap = answer_check_state.snapshot(retrieval)
    return tuple(snap.get(k) for k in _JUDGE_FIELDS)


def _running_as_saved(app_state: Any, retrieval: Any) -> bool:
    """Whether the running check is the one these settings ask for. A Save
    with nothing changed still repairs a daemon whose check never switched
    (4.1.19 saved a key without switching Jev on until a restart)."""
    if support.live_engine(app_state) is None:
        return True  # built from these settings when it starts
    running = judge_selection.backend_of(support.live_judge(app_state))
    wanted = support._resolve_uncached(retrieval, laya_runtime.detect(retrieval))
    return running == wanted


@router.post("/save")
def post_save(request: Request, body: SaveRequest):
    base._gate(request)
    try:
        app_state = request.app.state
        with support.MUTATION_LOCK:
            before = support.effective_retrieval()
            mode = body.mode or str(before.sufficiency_judge)
            refused = _refusal(mode, body, before)
            if refused:
                return base._error(refused)
            new_key = body.key.strip()
            if new_key:
                try:
                    judge_keys.JudgeKeyStore().set_key(body.provider, new_key)
                except ValueError as exc:
                    return base._error(str(exc))
                except judge_keys.JudgeKeyStoreError:
                    return base._error(support.KEY_PROBLEM)
                support.forget_test("jev")
            retrieval = support.save(_apply_form(body, mode))
            if new_key or _judge_fields(retrieval) != _judge_fields(before) \
                    or not _running_as_saved(app_state, retrieval):
                support.attach(app_state, retrieval)
            if not support.enforce(app_state, retrieval):
                return base._not_confirmed()
            status_body = support.build_status(app_state)
        status_body["message"] = "Saved. " + support.switched_message(mode, retrieval)
        return status_body
    except Exception:
        return base._internal_error()


@router.post("/laya/cancel")
def post_laya_cancel(request: Request):
    base._gate(request)
    if not laya_runtime.LayaInstallJob.instance().cancel():
        return base._error("No setup is running.", 409)
    return {"cancelling": True}


@router.post("/laya/test")
def post_laya_test(request: Request):
    """Run the on-device check once on the install that is set up, and keep
    the result. Explicit, never automatic: nothing tests in the background."""
    base._gate(request)
    try:
        if not sufficiency.laya_supported():
            return base._error("Needs a Mac with Apple Silicon.")
        with support.MUTATION_LOCK:
            if base._a_laya_job_is_running():
                return base._error("Wait for the current setup or check to finish.", 409)
            laya = laya_runtime.detect(support.effective_retrieval())
            if laya.state != laya_runtime.STATE_READY:
                return base._error("Set up the on-device check first, then test it.")
            before_verify, on_done = base._job_hooks(request.app.state)

            def finished(status: Any) -> None:
                ok = status.state == laya_runtime.STATE_READY
                seconds = float(status.step) if status.step.replace(".", "", 1).isdigit() else None
                support.record_test("laya", ok, "Answered correctly." if ok else status.error,
                                    seconds)
                on_done(status)

            started = laya_runtime.LayaTestJob.instance().start(
                laya.python, laya.hf_home, laya.model_path,
                on_done=finished, before_verify=before_verify)
            if not started:
                return base._error("A test is already running.", 409)
        return JSONResponse({"running": True}, status_code=202)
    except Exception:
        return base._internal_error()


__all__ = ["router"]
