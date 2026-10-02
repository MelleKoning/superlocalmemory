# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One plain sentence for when the online answer check sends memories out.

Modes A and B say nothing leaves the device. The online answer check can be
turned on in any mode, so while it is on, every place that makes that claim
adds this sentence instead of keeping a label that is no longer true.

"On" means what the person chose: the check set to the hosted option and
its consent given as the boolean True (a hand-edited "true" is not consent,
exactly as the check itself reads it). Stdlib only.
"""

from __future__ import annotations

from typing import Any

_PROVIDER_NAMES = {"typesafe": "TypeSafe", "openrouter": "OpenRouter"}


def online_check_on(retrieval: Any) -> bool:
    mode = str(getattr(retrieval, "sufficiency_judge", "") or "").strip().lower()
    return mode == "jev" and getattr(retrieval, "sufficiency_jev_consent", False) is True


def _reorder_count(retrieval: Any) -> int:
    if (getattr(retrieval, "sufficiency_jev_rerank", False) is not True
            or getattr(retrieval, "sufficiency_jev_rerank_consent", False) is not True):
        return 0
    try:
        from superlocalmemory.retrieval.jev_rerank import clamp_rerank_k

        return int(clamp_rerank_k(getattr(retrieval, "sufficiency_jev_rerank_k", None)))
    except Exception:  # noqa: BLE001 — wording only; never fail a status call
        return 0


def online_check_notice(retrieval: Any) -> str:
    """"" while the online answer check is off; else what it sends, and to whom."""
    if not online_check_on(retrieval):
        return ""
    provider = str(getattr(retrieval, "sufficiency_jev_provider", "") or "").strip().lower()
    name = _PROVIDER_NAMES.get(provider, "the provider you chose")
    reorder = _reorder_count(retrieval)
    extra = f", and up to {reorder} memories to reorder them" if reorder else ""
    return (
        f"The online answer check is on: each recall sends its question and "
        f"its top memories to {name}{extra}."
    )


__all__ = ["online_check_notice", "online_check_on"]
