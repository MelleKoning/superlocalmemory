"""Release contract: a pip install cannot land on a version with a known fix.

The CI audit reads ``uv.lock``. A user who runs ``pip install --upgrade`` keeps
whatever transitive versions already satisfy the declared ranges, so the lock
alone does not protect them. These checks read the declared requirements and
prove each version the audit flagged is no longer installable.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]

# Versions the dependency audit reported with a published fix. None of them
# may satisfy the requirements SuperLocalMemory declares.
_FLAGGED_VERSIONS = (
    ("accelerate", "1.14.0"),
    ("anyio", "4.13.0"),
    ("httpx2", "2.5.0"),
    ("nltk", "3.10.0"),
    ("pyjwt", "2.13.0"),
    ("sentence-transformers", "5.3.0"),
    ("setuptools", "81.0.0"),
    ("torch", "2.11.0"),
    ("transformers", "5.5.4"),
    ("urllib3", "2.7.0"),
)


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _requirements(lines: list[str]) -> dict[str, Requirement]:
    parsed = (Requirement(line) for line in lines)
    return {canonicalize_name(req.name): req for req in parsed}


def _exact_pin(req: Requirement) -> str:
    specs = list(req.specifier)
    assert len(specs) == 1 and specs[0].operator == "==", str(req)
    return specs[0].version


@pytest.mark.parametrize(("name", "version"), _FLAGGED_VERSIONS)
def test_runtime_requirements_exclude_each_flagged_version(
    name: str, version: str,
) -> None:
    runtime = _requirements(_project()["dependencies"])

    assert name in runtime, f"{name} needs a declared floor"
    assert not runtime[name].specifier.contains(version, prereleases=True)


def test_developer_test_client_excludes_flagged_version() -> None:
    dev = _requirements(_project()["optional-dependencies"]["dev"])

    assert not dev["httpx2"].specifier.contains("2.5.0", prereleases=True)


def test_search_extra_matches_the_runtime_pins() -> None:
    project = _project()
    runtime = _requirements(project["dependencies"])
    search = _requirements(project["optional-dependencies"]["search"])

    for name, req in search.items():
        assert _exact_pin(req) == _exact_pin(runtime[name]), name


def test_import_time_version_guard_matches_the_runtime_pins() -> None:
    source = (ROOT / "src" / "superlocalmemory" / "__init__.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    guard = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_REQUIRED_VERSIONS"
            for target in node.targets
        )
    )
    runtime = _requirements(_project()["dependencies"])

    for module, expected in guard.items():
        name = canonicalize_name(module.replace("_", "-"))
        assert expected == _exact_pin(runtime[name]), module
