# Web access architecture

Web access adds an optional, authenticated web connection to [the existing local
engine](../ARCHITECTURE.md). Nothing about local use changes when it is off.

![Web MCP clients use HTTPS and OAuth; Cloudflare checks grants and forwards through an authenticated outbound laptop relay](assets/remote-gateway-boundaries.svg)

## One engine, two access paths

Local Claude Code, Codex, the CLI, SDKs and MCP clients keep their existing entry
points and their full tool set. A web app reaches the same loopback MCP entry
point through a companion that runs inside SLM's own Python runtime, so there is
nothing for you to install. Both paths use the same governed admission, the same
write coordination and the same canonical database. The companion never opens a
second writer.

| Component | Responsibility | Boundary |
|---|---|---|
| Dashboard, **Connected apps** | Start enrollment, show consent and state, remove an app, turn Web access off | Local installation authentication; no cloud secrets in browser storage |
| Local enrollment service | Journal the opt-in and each operation so a retry never duplicates; protect the credential; run the companion | Its own state, separate from the memory database |
| Companion | Hold an outbound WebSocket to the gateway and forward admitted requests to a fixed loopback MCP URL | No caller-chosen destination, no redirects, bounded buffers |
| Public MCP endpoint | HTTPS MCP resource, token verification, and listing only the tools the caller may use | Every request is authenticated before routing |
| Authorization service and registry | Bind owner, app, computer, profile and permissions; enforce expiry and revocation | The server is the authority; a flag in the browser is not permission |
| Relay | Pair each admitted request with the computer's live connection, with a deadline | Holds no memory database |
| SLM engine and storage | Memory tools, governed writes, recall | Existing local permissions stay authoritative |

The public side uses HTTPS and OAuth. The link between the gateway and your
computer is a private outbound WebSocket, which is why no inbound port, DNS
record or tunnel is needed on your side.

## A web call

1. You turn on Web access in the dashboard for the active profile and choose
   what apps may do.
2. The app signs in through GitHub and receives only the permissions you
   approved. A permission can never exceed that ceiling.
3. The gateway checks the token, the app, the connection and its current state
   on every request. The effective permissions are the intersection of the
   token's scope, the original consent and the connection's current policy.
4. The gateway accepts only these tools: `recall`, `search`, `fetch` and
   `get_status` for read; `remember` for save; `session_init`, `close_session`,
   `report_feedback` and `report_outcome` for session tools. Everything else is
   refused, and the app's tool list shows only what it may call.
5. The relay hands the request to your computer's live connection with a
   25-second deadline. A request body is limited to 1 MiB and a response to 4 MiB.
   The relay holds at most 8 unanswered requests per computer.
6. The companion forwards the original request to the local MCP endpoint with
   the local installation credential. Cloud credentials never replace it.
7. The engine runs it through its normal admission and write path, and the answer
   returns the same way.

A lost response is not proof that a write failed, and a reconnect never replays
a write that may have completed. A save carries an `idempotency_key`, and a
retry with the same key returns the first result.

## Data and availability

Your database stays on your computer, but request and response contents pass
through the gateway and the AI service you connected. The gateway keeps identity,
consent and connection records plus operational metadata; it does not keep a copy
of your memories, and request logging is off by default so memory text and OAuth
parameters are not recorded.

A sleeping computer, a lost network or a stopped companion makes Web access
unavailable. The gateway treats the computer as asleep after 45 seconds without
a heartbeat and answers `connector_asleep`; that is never presented as an empty
recall. A gateway or sign-in failure affects only Web access.

The computer's credential lasts 30 days and is renewed after the halfway point.
An app's own sign-in slides forward each time it is used and lapses after 30
days idle. The daily allowance is a count of tool calls per day, reset at
midnight UTC.

The provider manages the shared endpoint hostnames. You never need a Cloudflare
account, a named tunnel or `cloudflared`.

## Visual assets

The [integrated SVG](assets/slm-integrated-architecture.svg), [connection
overview SVG](assets/slm-local-and-remote.svg), [mobile
SVG](assets/slm-local-and-remote-mobile.svg) and [boundary
SVG](assets/remote-gateway-boundaries.svg) are editable source diagrams. The PNG
files are renders of them.
