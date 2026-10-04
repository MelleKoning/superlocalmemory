@echo off
:: slm-launch.bat — WP-F SuperLocalMemory MCP launcher (Windows)
::
:: Cross-platform counterpart: slm-launch (POSIX bash)
::
:: NOT reached through plugin/.mcp.json. That file names the extensionless
:: "${CLAUDE_PLUGIN_ROOT}/scripts/slm-launch", and a host that spawns it without
:: a shell (CreateProcess, Node spawn without shell) never resolves it to this
:: .bat: CreateProcess cannot start a batch file and only appends .exe to a
:: name without an extension. This file runs only when started through cmd.exe
:: (for example a hand-written MCP entry using "cmd /c"). It does not read
:: SLM_LAUNCHER; see MCP_JSON_NOTES.md and issue #139. Since 4.1.20 the plugin's
:: .mcp.json starts the INSTALLED slm on Windows through %ComSpec% instead.
:: Proven on the Windows CI runner: tests/test_plugin/test_windows_mcp_spawn_premise.py
::
:: Resolves the correct venv binary for Windows and joins the namespace daemon
:: before opening MCP, preserving one writer for parallel Claude sessions.
::
:: On Windows, Python venv places entry points in Scripts\ (not bin\ like POSIX).
:: This launcher bridges the path difference so ONE .mcp.json command field works
:: cross-platform.
::
:: Environment:
::   CLAUDE_PLUGIN_DATA — persistent data dir where venv lives

"%CLAUDE_PLUGIN_DATA%\venv\Scripts\slm.exe" serve start 1>&2
if errorlevel 1 (
    echo SLM plugin: unable to start the owned daemon; refusing a direct MCP writer. 1>&2
    exit /b 1
)

"%CLAUDE_PLUGIN_DATA%\venv\Scripts\slm.exe" mcp
