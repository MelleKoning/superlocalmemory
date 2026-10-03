@echo off
REM SuperLocalMemory V3 - Windows CLI Wrapper
REM Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
REM Licensed under MIT License
REM Repository: https://github.com/qualixar/superlocalmemory

setlocal enabledelayedexpansion

REM Resolve the package src/ directory for PYTHONPATH
set "SLM_PKG_DIR=%~dp0..\src"

REM Handle --version / -v directly (fast path, no Python needed)
if "%~1"=="--version" goto :show_version
if "%~1"=="-v" goto :show_version

REM LLD-06 §6.3 — prefer the PyInstaller-built binary on the hot hook path.
REM Resolve data root honoring the SLM_DATA_DIR -> SL_MEMORY_PATH -> SLM_HOME
REM cascade (mirrors infra/data_root.py). Highest precedence is applied last.
set "_SLM_ROOT=%USERPROFILE%\.superlocalmemory"
if defined SLM_HOME set "_SLM_ROOT=%SLM_HOME%"
if defined SL_MEMORY_PATH set "_SLM_ROOT=%SL_MEMORY_PATH%"
if defined SLM_DATA_DIR set "_SLM_ROOT=%SLM_DATA_DIR%"
set "SLM_HOOK_BIN=%_SLM_ROOT%\bin\slm-hook\slm-hook.exe"
if defined SLM_HOOK_BINARY set "SLM_HOOK_BIN=%SLM_HOOK_BINARY%"
if "%~1"=="hook" if "%~2"=="user_prompt_submit" (
    if exist "%SLM_HOOK_BIN%" (
        if not "%SLM_HOOK_BINARY_DISABLED%"=="1" (
            "%SLM_HOOK_BIN%"
            exit /b %ERRORLEVEL%
        )
    )
)

REM Find Python 3
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3
    goto :version_check
)
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :version_check
)
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3
    goto :version_check
)

echo Error: Python 3.12+ not found.
echo Install from: https://python.org/downloads/
exit /b 1

:version_check
REM L3-22: this is the repository-clone launcher — the npm-packaged runtime
REM goes through its own .slm-venv, guaranteed 3.12+ by scripts/postinstall.js.
REM Without this check, a `python`/`python3`/`py` on PATH that is too old ran
REM anyway and failed later with a cryptic error instead of one clear message.
set "PY_VER_STR="
set "PY_MAJOR="
set "PY_MINOR="
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set "PY_VER_STR=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PY_VER_STR%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if not defined PY_MAJOR goto :version_unsupported
if %PY_MAJOR% LSS 3 goto :version_unsupported
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 12 goto :version_unsupported
goto :run

:version_unsupported
echo Error: Python 3.12+ required ^(found: %PY_VER_STR%^ via %PYTHON_CMD%^). Install from https://python.org/downloads/ 1>&2
exit /b 1

:show_version
REM Read version from package.json via findstr
for /f "tokens=2 delims=:," %%a in ('findstr /C:"\"version\"" "%~dp0..\package.json"') do (
    set "VER=%%~a"
    set "VER=!VER: =!"
    echo superlocalmemory !VER!
    exit /b 0
)
echo superlocalmemory unknown
exit /b 0

:run
REM Set PYTHONPATH so Python finds the npm package's src/ directory
if defined PYTHONPATH (
    set "PYTHONPATH=%SLM_PKG_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%SLM_PKG_DIR%"
)

REM Prevent PyTorch Metal/MPS GPU memory reservation
set "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
set "PYTORCH_MPS_MEM_LIMIT=0"
set "PYTORCH_ENABLE_MPS_FALLBACK=1"
set "TOKENIZERS_PARALLELISM=false"
set "TORCH_DEVICE=cpu"
set "CUDA_VISIBLE_DEVICES="

%PYTHON_CMD% -m superlocalmemory.cli.main %*
exit /b %ERRORLEVEL%
