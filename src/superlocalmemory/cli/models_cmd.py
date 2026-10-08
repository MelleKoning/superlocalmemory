# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm models`` — what SLM would recommend on THIS computer, and why (#4.1.22).

Varun: people running Ollama locally can pull a more powerful model than the
old hardcoded default, but only if they know it exists and that it fits their
machine. This is a read-only report over the same catalogue and ranking the
setup wizard and the dashboard use (``core.model_catalog``), so the three
surfaces never disagree. It adds no request of its own: Ollama's installed
models are read through ``cli.setup_wizard._ollama_installed_models`` (the
one reviewed ``GET /api/tags`` call already covered by the outbound gate).
"""

from __future__ import annotations

from argparse import Namespace


def cmd_models(args: Namespace) -> None:
    """Print installed + recommended local models, and the hosted catalogue."""
    from superlocalmemory.cli.setup_wizard import _ollama_installed_models
    from superlocalmemory.core import model_catalog

    ram_gb = _get_ram_gb()
    names = _ollama_installed_models() or []
    recs = model_catalog.recommend_local_llms(ram_gb or None, names)

    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        data = model_catalog.catalog()
        data["machine"] = {"ram_gb": ram_gb}
        data["installed"] = names
        data["local_recommendations"] = [_rec_as_dict(rec) for rec in recs]
        json_print("models", data=data)
        return

    print(f"This computer: {ram_gb:.1f} GB")
    print()
    print("Installed Ollama models")
    if not recs:
        print("  (none found — is Ollama running? `ollama serve`)")
    for rec in recs:
        marker = " (installed)" if rec.installed else ""
        print(f"  {rec.model_id} — {rec.reason}{marker}")
    print()
    print("Hosted (Mode C)")
    for entry in model_catalog.HOSTED_LLMS:
        print(f"  {entry.id} — {entry.price} — {entry.advice}")


def _get_ram_gb() -> float:
    from superlocalmemory.core.machine import total_ram_gb

    return total_ram_gb()


def _rec_as_dict(rec) -> dict:
    from superlocalmemory.core import model_catalog

    return {
        "model_id": rec.model_id,
        "installed": rec.installed,
        "fits": rec.fits,
        "reason": rec.reason,
        "entry": model_catalog.as_dict(rec.entry) if rec.entry is not None else None,
    }


__all__ = ["cmd_models"]
