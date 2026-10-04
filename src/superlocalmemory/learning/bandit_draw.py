# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The Thompson draw, keyed so one question over one posterior is one answer.

Before 4.1.20 every ``choose`` drew fresh entropy from ``secrets.SystemRandom``.
The arm it picked sets the channel weights, and those weights decide the order,
so the SAME question over the SAME memories came back in different orders —
measured 5 orders in 40 identical recalls, the top answer changing 14 times.
That also defeated the answer-check memo, whose key is the order it read.

The draw is now a pure function of what the choice is about:

    seed = HMAC(install key, profile | stratum | normalised question | posterior)

* Same question, same posterior -> same arm. Repeatable.
* A different question -> an independent draw, so across the questions people
  actually ask the policy is still Thompson sampling and still explores.
* A settled reward changes the posterior, which changes the seed, so learning
  reaches repeat questions too — the answer moves when the evidence moves.
* Keyed by the install's secret (its own derived key, never the token itself),
  so the draw is no more predictable to someone without that secret than the
  system RNG it replaces.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import secrets
from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Domain separation: the install token also signs fact markers, so the draw
#: uses a key derived for this purpose only.
_DRAW_KEY_LABEL = b"slm:bandit-draw:v1"

#: Used only when the install key cannot be read. Stable for the life of the
#: process, so a recall is still repeatable within it.
_PROCESS_FALLBACK_KEY = secrets.token_bytes(32)


def _draw_key() -> bytes:
    """The install-derived key for the draw. Never raises."""
    try:
        from superlocalmemory.core.security_primitives import ensure_install_token

        token = ensure_install_token()
    except Exception as exc:  # pragma: no cover — token I/O failure
        logger.debug("bandit draw: install key unavailable (%s)", type(exc).__name__)
        return _PROCESS_FALLBACK_KEY
    return hmac.new(token.encode("utf-8"), _DRAW_KEY_LABEL, hashlib.sha256).digest()


def normalize_question(text: str | None) -> str:
    """Case- and whitespace-insensitive form, so trivially re-typed asks match."""
    return " ".join(str(text or "").casefold().split())


def posterior_digest(posteriors: Mapping[str, tuple[float, float]]) -> str:
    """A version stamp of the stratum's learned state: changes on every update."""
    material = json.dumps(
        sorted((str(arm), repr(float(a)), repr(float(b)))
               for arm, (a, b) in posteriors.items()),
        ensure_ascii=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def draw_rng(
    profile_id: str,
    stratum: str,
    question: str,
    posteriors: Mapping[str, tuple[float, float]],
) -> random.Random:
    """A generator seeded by everything the choice depends on, and nothing else."""
    material = json.dumps(
        [str(profile_id), str(stratum), normalize_question(question),
         posterior_digest(posteriors)],
        ensure_ascii=True,
    )
    seed = hmac.new(_draw_key(), material.encode("utf-8"), hashlib.sha256).digest()
    return random.Random(int.from_bytes(seed, "big"))


__all__ = ["draw_rng", "normalize_question", "posterior_digest"]
