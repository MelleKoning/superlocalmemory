# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One-time reading of settings saved by older releases (used by core/config.py).

Kept out of config.py so the upgrade rules live in one small, testable place.
Stdlib only.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("superlocalmemory.core.config")

#: The ``math`` keys every 4.1.0-4.1.17 save() wrote, all eleven, every time
#: (checked against the v4.1.0, v4.1.5, v4.1.10 and v4.1.17 tags). A section
#: holding all of them without ``sheaf_default_reviewed`` is that old dump, whose
#: ``sheaf_at_encoding: true`` is a stored default. A section missing any of them
#: was written by a person, and what it says is a choice.
PRE_4_1_18_MATH_DUMP_KEYS = frozenset({
    "fisher_temperature", "fisher_bayesian_update", "fisher_mode",
    "langevin_dt", "langevin_temperature", "langevin_persist_positions",
    "langevin_weight_range", "ebbinghaus_langevin_coupling_enabled",
    "sheaf_at_encoding", "sheaf_contradiction_threshold", "sheaf_max_edges_per_check",
})

_sheaf_switch_off_reported = False


def is_saved_math_dump(section: dict) -> bool:
    return PRE_4_1_18_MATH_DUMP_KEYS <= set(section)


def sheaf_reviewed(raw_section: dict, fields: dict) -> dict:
    """``fields`` (the math section being loaded) with the 4.1.18 default applied:
    an old full dump without the review marker reads as off, once, said aloud;
    anything a person wrote is kept as written. Returns a new dict."""
    if "sheaf_default_reviewed" in raw_section or not is_saved_math_dump(raw_section):
        return dict(fields)
    if fields.get("sheaf_at_encoding") is True:
        report_sheaf_switch_off()
    return {**fields, "sheaf_at_encoding": False}


def report_sheaf_switch_off() -> None:
    """Say once per process that the old stored default was switched off.

    Once: until something saves the config (which records the review), every
    load reads the same old file, and a warning per `slm` command is noise.
    """
    global _sheaf_switch_off_reported
    if _sheaf_switch_off_reported:
        return
    _sheaf_switch_off_reported = True
    logger.warning(
        "The store-time consistency check (math.sheaf_at_encoding) is now off by "
        "default; this config held the old default (true), so it is off. To keep "
        "it on, set math.sheaf_at_encoding to true and math.sheaf_default_reviewed "
        "to true in config.json.")


__all__ = ["PRE_4_1_18_MATH_DUMP_KEYS", "is_saved_math_dump", "report_sheaf_switch_off",
           "sheaf_reviewed"]
