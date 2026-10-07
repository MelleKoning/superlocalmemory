# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Why the owned daemon cannot be used, in actionable terms.

Moved out of ``cli/daemon.py`` (4.1.22) so that file stops growing; the public
names are re-exported there. Every daemon helper is looked up on
``superlocalmemory.cli.daemon`` at call time, so tests that patch that module
keep steering the diagnosis.
"""

from __future__ import annotations

_GENERIC_UNAVAILABLE = {
    "reason": "unknown",
    "message": "Owned daemon is unavailable; retry later.",
    "hint": "Run `slm doctor`, then `slm restart` if it stays down.",
}

_LIVENESS_DIAGNOSIS = {
    "process_exited": (
        "daemon_process_exited",
        "the recorded daemon process (pid {pid}) is no longer running",
        "Start it again with `slm serve start`.",
    ),
    "process_zombie": (
        "daemon_process_exited",
        "the recorded daemon process (pid {pid}) has exited and is awaiting "
        "reaping by its parent",
        "Start it again with `slm serve start`.",
    ),
    "process_unreadable": (
        "daemon_process_unreadable",
        "the recorded daemon process (pid {pid}) could not be inspected; it "
        "may belong to another user",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
    "start_token_mismatch": (
        "pid_reused_by_another_process",
        "pid {pid} is alive but is a different process than the daemon that "
        "wrote {path}; the daemon exited and its pid was recycled",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
    "identity_mismatch": (
        "daemon_identity_mismatch",
        "pid {pid} did not match the process identity recorded in {path} and "
        "the process on port {port} did not prove it owns that identity; the "
        "recorded creation time can also diverge on its own if this machine's "
        "clock is stepped (common under WSL2)",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
}


def describe(d) -> dict[str, str]:
    """Body of ``daemon._describe_daemon_unavailability`` (``d`` = that module)."""
    from superlocalmemory.cli import daemon_startup

    path = d.descriptor_path()
    descriptor = d.read_descriptor()
    if descriptor is None:
        if path.exists():
            return {
                "reason": "descriptor_unusable",
                "message": (
                    f"{path} is unreadable, malformed, or belongs to another "
                    f"data root or user."
                ),
                "hint": "Run `slm restart` to publish a fresh descriptor.",
            }
        if (
            daemon_startup.this_process_is_spawning()
            or daemon_startup.lock_is_held_by_another_process()
        ):
            return daemon_startup.starting_diagnosis(None)
        if d._verified_legacy_health() is not None:
            return {
                "reason": "legacy_daemon_request_failed",
                "message": (
                    "a pre-descriptor daemon answered health but rejected or "
                    "dropped the request."
                ),
                "hint": "Run `slm restart` to upgrade it to an owned daemon.",
            }
        return {
            "reason": "no_daemon",
            "message": f"no daemon is registered for this data root ({path} is absent).",
            "hint": "Run `slm serve start`.",
        }

    alive, evidence = d._resolve_descriptor_liveness(descriptor)
    if not alive:
        reason, template, hint = _LIVENESS_DIAGNOSIS.get(
            evidence,
            (
                "daemon_identity_mismatch",
                "pid {pid} did not match the identity recorded in {path}",
                "Run `slm restart` to publish a fresh descriptor.",
            ),
        )
        return {
            "reason": reason,
            "message": template.format(
                pid=descriptor.pid, port=descriptor.port, path=path,
            ) + ".",
            "hint": hint,
        }

    starting = getattr(descriptor, "state", "") == "starting"
    probe_timeout = daemon_startup.health_probe_timeout(descriptor)
    health = daemon_startup.probe_health(d, descriptor.port, probe_timeout)
    if health is None:
        if starting:
            return daemon_startup.starting_diagnosis(descriptor)
        return {
            "reason": "daemon_unreachable",
            "message": (
                f"the owned daemon (pid {descriptor.pid}) is running but did "
                f"not answer http://127.0.0.1:{descriptor.port}/health within "
                f"{probe_timeout:g}s."
            ),
            "hint": (
                f"Check {d.state_path('logs', 'daemon.log')} for a stalled "
                "request, or run `slm restart` if it stays unresponsive."
            ),
        }
    if not d.descriptor_matches_health(descriptor, health):
        return {
            "reason": "port_owned_by_another_daemon",
            "message": (
                f"port {descriptor.port} answered health but with a different "
                f"daemon identity than {path} records."
            ),
            "hint": (
                "Another SuperLocalMemory instance holds that port. Stop it, "
                "or set SLM_DAEMON_PORT to a free port."
            ),
        }
    return {
        "reason": "request_rejected",
        "message": (
            f"the owned daemon (pid {descriptor.pid}) is healthy but rejected "
            f"or dropped this request."
        ),
        "hint": f"Check {d.state_path('logs', 'daemon.log')} for the failing request.",
    }
