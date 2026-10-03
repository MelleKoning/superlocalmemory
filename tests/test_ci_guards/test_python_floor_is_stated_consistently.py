# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""pyproject.toml has required Python >=3.12,<3.15 since the patched
cryptography runtime shipped for V4. Two places still quoted the old,
pre-V4 floor: the LangChain integration's PyPI classifiers (3.10, missing
3.14) and docs/troubleshooting.md's "Python not found" guidance, which told
a reader needing a fix that 3.10 was fine (L3-23).
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - repo floor is 3.12, kept for a stray old runner
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGCHAIN_PYPROJECT = REPO_ROOT / "ide" / "integrations" / "langchain" / "pyproject.toml"
TROUBLESHOOTING = REPO_ROOT / "docs" / "troubleshooting.md"


def test_langchain_classifiers_match_requires_python() -> None:
    with LANGCHAIN_PYPROJECT.open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["project"]["requires-python"] == ">=3.12,<3.15"

    classifiers = metadata["project"]["classifiers"]
    versioned = {
        c.rsplit(" :: ", 1)[-1]
        for c in classifiers
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert versioned == {"3.12", "3.13", "3.14"}, (
        f"LangChain integration classifiers say {sorted(versioned)}; "
        f"requires-python is {metadata['project']['requires-python']!r}"
    )


def test_troubleshooting_states_the_real_python_floor() -> None:
    text = TROUBLESHOOTING.read_text(encoding="utf-8")
    assert "Python 3.10 or later" not in text, (
        "docs/troubleshooting.md still tells a reader 3.10 is enough; the "
        "real floor (pyproject.toml) is 3.12"
    )
    assert "Python 3.12 or later" in text
