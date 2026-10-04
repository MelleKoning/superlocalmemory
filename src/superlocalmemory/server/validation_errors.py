# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""A refused request body is described, never repeated back.

FastAPI's default answer to a body that fails validation (422) quotes the
offending value in ``input``. For the routes that take a secret, that value is
the secret: a provider key one character too long came back in full in the
response to ``POST /api/v3/answer-check/jev/key``, where browser tools, HAR
exports and any proxy log would keep it. The same held for passwords and API
keys on every other route.

This handler keeps what a caller needs to fix the request (where, what kind
of error, the message, and the rule it broke) and drops the value itself.
"""

from __future__ import annotations

from typing import Any

#: Error fields that can carry the submitted value.
_VALUE_FIELDS = frozenset({"input", "url"})


def describe_errors(errors: Any) -> list[dict[str, Any]]:
    """Validation errors without the values that failed."""
    from fastapi.encoders import jsonable_encoder

    described = []
    for error in errors or ():
        if isinstance(error, dict):
            described.append({k: v for k, v in error.items() if k not in _VALUE_FIELDS})
    return jsonable_encoder(described)


async def validation_error_handler(_request: Any, exc: Any):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": describe_errors(exc.errors())})


def install(application: Any) -> None:
    """Answer every 422 on ``application`` without quoting the request."""
    from fastapi.exceptions import RequestValidationError

    application.add_exception_handler(RequestValidationError, validation_error_handler)


__all__ = ["describe_errors", "install", "validation_error_handler"]
