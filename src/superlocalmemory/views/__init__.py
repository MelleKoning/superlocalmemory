# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Saved views: named, profile-scoped recall queries (issue #113).

A view is a name plus a recall query and a few of recall's own filters. Running
it is running recall, so a view ranks, checks and repeats exactly like the
query it stands for, and every result carries the id of its memory. No prompt
is ever run over the store.

    model   what a valid view is (validation, refusal codes, filter registry)
    store   the ``saved_views`` table in learning.db (migration M054)
    runner  view -> recall arguments, and recall's answer -> a view run
"""

from superlocalmemory.views.model import SavedView, ViewError
from superlocalmemory.views.runner import recall_arguments, shape_run, view_session_id
from superlocalmemory.views.store import ViewStore, default_store

__all__ = ["SavedView", "ViewError", "ViewStore", "default_store", "recall_arguments",
           "shape_run", "view_session_id"]
