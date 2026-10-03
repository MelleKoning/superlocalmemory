# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""GitHub release rotation for cloud backups.

Each backup is one GitHub release. Only the newest ``MAX_GITHUB_RELEASES`` are
kept, which is also what removes unencrypted releases left by 4.1.18 and
earlier: they are rotated out as encrypted backups replace them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("superlocalmemory.cloud_backup")


MAX_GITHUB_RELEASES = 5  # Keep last 5 backups, delete older ones


def _cleanup_old_releases(full_repo: str, headers: dict) -> None:
    """Delete old GitHub releases to prevent repo storage from exploding.

    Keeps the most recent MAX_GITHUB_RELEASES releases and deletes the rest.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{full_repo}/releases",
            headers=headers,
            params={"per_page": 100},
            timeout=15,
        )
        if resp.status_code != 200:
            return

        releases = resp.json()
        if len(releases) <= MAX_GITHUB_RELEASES:
            return

        # Sort by creation date (newest first), delete everything after MAX
        releases.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        to_delete = releases[MAX_GITHUB_RELEASES:]

        for release in to_delete:
            release_id = release["id"]
            tag = release.get("tag_name", "")

            # Delete release
            httpx.delete(
                f"https://api.github.com/repos/{full_repo}/releases/{release_id}",
                headers=headers,
                timeout=15,
            )
            # Delete the tag too (releases leave orphan tags)
            httpx.delete(
                f"https://api.github.com/repos/{full_repo}/git/refs/tags/{tag}",
                headers=headers,
                timeout=15,
            )
            logger.info("Cleaned up old release: %s", tag)

        logger.info("GitHub cleanup: removed %d old releases, kept %d", len(to_delete), MAX_GITHUB_RELEASES)

    except Exception as exc:
        logger.warning("GitHub release cleanup failed (non-critical): %s", exc)
