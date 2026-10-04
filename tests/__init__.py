# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Any test module under tests/ imports this package first, so the live-root
isolation is installed even when neither the root conftest nor pyproject's
addopts were loaded (``--noconftest -o addopts=``). Outside pytest it is inert."""

import sys

if "_pytest" in sys.modules:
    import tests._isolation_plugin  # noqa: F401
