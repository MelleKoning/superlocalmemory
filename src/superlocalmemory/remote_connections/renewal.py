"""When the laptop renews its gateway credential.

The gateway issues a credential for 30 days and accepts a renewal only in its
second half (integrations/remote-gateway/src/credential-lifetime.ts; a Node test
keeps the window below equal to it).
"""

from __future__ import annotations

RENEWAL_WINDOW_MS = 15 * 24 * 3600 * 1000
#: Waits are capped so the wall clock is read again at least hourly. asyncio's
#: clock does not advance while the computer sleeps, so one long wait could
#: overshoot the due time by days.
RENEWAL_CHECK_S = 3600.0
#: After an outage, try again this soon; there are 15 days of margin.
RENEWAL_RETRY_S = 900.0


def renewal_delay_s(expires_at_ms: int, now_ms: float) -> float:
    """Seconds to wait before checking again; 0 means renew now."""
    due = expires_at_ms - RENEWAL_WINDOW_MS
    if now_ms >= due:
        return 0.0
    return min((due - now_ms) / 1000.0, RENEWAL_CHECK_S)
