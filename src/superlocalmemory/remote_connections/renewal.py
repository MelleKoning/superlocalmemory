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


#: Renewal starts with 15 days left, so under 7 days left means it has been
#: failing for over a week: tell the owner before access actually stops.
ENDING_SOON_MS = 7 * 24 * 3600 * 1000


def access_state(expires_at_ms: int, now_ms: float, transport_state: str | None) -> str:
    """Owner-facing Web access status for a completed connection.

    One of renews_automatically, ending_soon, ended, sign_in_required. A lapsed
    or removed sign-in outranks the expiry, because only signing in again helps.
    """
    if transport_state == "authorization_required":
        return "sign_in_required"
    remaining = expires_at_ms - now_ms
    if remaining <= 0:
        return "ended"
    if remaining <= ENDING_SOON_MS:
        return "ending_soon"
    return "renews_automatically"
