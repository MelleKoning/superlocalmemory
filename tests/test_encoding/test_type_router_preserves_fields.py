# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Routing a fact to its typed store changes its type and nothing else.

It used to rebuild each fact field by field, silently resetting every field
the list forgot — sharing, pinning, lifecycle and now the kind a caller or the
extractor attached — whenever the dataclass grew.
"""

from __future__ import annotations

import dataclasses

from superlocalmemory.encoding.type_router import TypeRouter
from superlocalmemory.storage.memory_kinds import KindSource, MemoryKind
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryLifecycle, Mode


def test_route_facts_keeps_scope_shared_with_pinned_lifecycle_access_count_and_kind() -> None:
    fact = AtomicFact(
        content="The billing service stores money as integer cents in Postgres.",
        fact_type=FactType.OPINION,
        profile_id="work", scope="shared", shared_with=["team"], pinned=True,
        lifecycle=MemoryLifecycle.WARM, langevin_position=[0.1, 0.2], access_count=7,
        memory_kind=MemoryKind.OPINION.value, memory_kind_source=KindSource.CALLER.value,
        memory_kind_confidence=None, memory_kind_recipe="caller",
        memory_kind_at="2026-10-03T00:00:00+00:00",
    )
    [routed] = TypeRouter(mode=Mode.A).route_facts([fact])
    assert routed.fact_type is FactType.SEMANTIC, "the type itself is re-routed"
    expected = dataclasses.replace(fact, fact_type=routed.fact_type)
    assert dataclasses.asdict(routed) == dataclasses.asdict(expected)
    assert routed is not fact and fact.fact_type is FactType.OPINION
