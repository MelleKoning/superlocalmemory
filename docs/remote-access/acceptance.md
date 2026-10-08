# Checking a connection

Use this to confirm that an app really works with your memory, end to end. A
generic "supports MCP" claim is not evidence; only a completed round trip inside
the app is. Do the checks with a synthetic memory you do not mind losing.

## Check one app

1. In the dashboard, open **Connected apps** and set the app up as described in
   [Connect an app](onboarding.md). Confirm the page says **Web access is on**.
2. In the app, confirm it lists only the tools its permissions allow: `recall`,
   `search`, `fetch` and `get_status`, plus `remember` if saving was allowed and
   the session tools if those were allowed.
3. Save a harmless unique sentence on your computer with `slm remember "web
   access check 7f3a"`. In the app, ask for it by meaning. Compare what the
   app reports with `slm recall "web access check 7f3a" --json`, not just the
   assistant's paraphrase.
4. If you allowed saving, ask the app to remember a different unique sentence.
   Then run `slm recall` for it on your computer. Ask the app again in a fresh
   conversation.
5. Remove the app's access under **Your connected apps**. Ask it to recall
   again: it must be refused (`REVOKED`).
6. Put your computer to sleep for a minute, then ask the app. It should report
   that the computer is asleep or offline (`connector_asleep` or
   `connector_offline`), not an empty answer. Wake the computer and ask again.

## What to record

Note which app and plan you tested, the SLM version (`slm --version`), the time
each step took, and what the app actually returned. Keep memory text and tokens
out of anything you share. Report a result that is slow, empty, abstained,
unavailable or timed out as that, not as one outcome: a correct answer that takes
a few seconds is slow, not broken, and the time includes sign-in and the app's own
overhead as well as SLM.

## What the checks cover

| Area | What you are confirming |
|---|---|
| Local use | `slm recall`, hooks, MCP clients and mesh work with Web access off, on and revoked |
| Sign-in | GitHub sign-in completes; an app only gets the permissions you ticked |
| Read | `recall`, `search` and `fetch` return what your computer holds |
| Save | `remember` stores once, even when retried with the same `idempotency_key` |
| Refusals | A read-only app cannot save; any app is refused after removal |
| Availability | A sleeping or offline computer is reported honestly and local SLM is unaffected |
