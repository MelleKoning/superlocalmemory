# Connect an app

Web access is off until you turn it on, and it only ever turns on when you
approve an app. SLM works fully on your computer without it. See [Web
access](README.md) for what it is and what an app can do.

## Steps

1. Open the dashboard with `slm dashboard`. In the sidebar, under **Integrations**,
   choose **Connected apps**. The page says **Internet access is off** until you
   approve an app.
2. Under **Set up an app**, choose the app: ChatGPT, Claude (web), Claude Code
   (web), Composio, Muse or **Other app (MCP)**. The choice only picks the
   instructions shown later. Whether your account with that app allows a custom
   remote MCP connection is up to the app.
3. Under **What this app can do**, tick **Turn on internet access for this app**.
   Reading your memories is always included. **Allow saving memories** lets the
   app remember new things; leave it off to keep the app read-only. **Allow
   session tools** lets it open and close memory sessions. Saving and session
   tools are separate choices.
4. Select **Link this computer with GitHub**. A GitHub sign-in page opens; if
   your browser blocks it, use **Continue sign-in**. SLM starts the connection
   from this computer itself; there is nothing else to install and no Cloudflare
   account, DNS record or tunnel to set up. Sign-in links that expire can be
   renewed with **Restart sign-in**, which keeps the permissions you chose.
5. Wait for the page to say **Web access is on**. It confirms that this computer
   is linked and that the connection works from here.
6. Under **Add SuperLocalMemory to** and the app's name, follow the steps for your app. Most apps only
   ask for the **MCP server URL**; some also ask for the **OAuth metadata URL**.
   Both are shown on the page and under **Technical details**.
   - Composio: add a Custom MCP named SuperLocalMemory, paste the server URL,
     choose OAuth, and paste the OAuth metadata URL in Advanced settings.
   - Muse: it uses a private adapter and its secure OAuth connector. Give it
     the two URLs and approve access with the same GitHub account. Never paste
     tokens into a chat.
   - Other apps: add a remote MCP connector (some call it a custom connector),
     paste the server URL, choose OAuth, and approve this memory profile.
7. Sign in with the same GitHub account when the app asks, and approve the
   permissions. Paste the text from **Copy instructions** into the app's
   instructions so it knows when to recall and what to save (see [Web
   agents](../web-agents/README.md)).
8. Ask the app something your memory knows. Only a completed round trip proves
   the app is connected. If you allowed saving, test it with a harmless
   sentence and ask for it back a few seconds later.

The page reaches one memory profile, the profile active when you turn Web access
on, and shows it under **Memory profile**. Switching profile clears an
unfinished sign-in and resets the permission boxes.

## Seeing and removing access

**Your connected apps** lists every app authorized on this computer, with what
it can do (Read, Save, Session tools), when it connected and when it was last
used. **Remove access** ends that one app after you confirm. **Refresh** reloads
the list. **Turn off** (or **Cancel setup**, while a connection is still
pending) ends Web access for all apps, after you confirm; your memory is
unchanged and you can turn it on again. Finished and cancelled connections stay
listed under **Past connections**.

## What the page tells you

| Message | Meaning |
|---|---|
| Internet access is off | Nothing is connected. Local use is unaffected |
| Waiting for GitHub sign-in | Finish the page that opened, or retry the same request |
| GitHub sign-in is done. Checking the connection | This computer is confirming the route |
| Web access is on | Apps you approve can reach your memory while this computer is online |
| Renews automatically | Nothing to do. The credential lasts 30 days and renews itself in its second half |
| Web access ends on (date) unless this computer reconnects | Renewal has been failing for over a week. Check that the computer is online |
| Web access has ended | Turn it on again to reconnect your apps |
| Sign in again to keep Web access working | The GitHub sign-in lapsed. Sign in again |
| A sign-in link expired | Use **Restart sign-in**; your permissions are kept |
| Connected apps are not available on this computer right now | The local service is not running or not reachable. Run `slm status`, then `slm restart` |

If an app reports an error code, see [Troubleshooting](../troubleshooting.md#web-access).

## What stays untouched

Turning Web access on does not change your modes, profiles, providers, hooks,
local MCP setup, mesh or database. The connection keeps its own state, and a
problem in it never stops local SLM from starting.
