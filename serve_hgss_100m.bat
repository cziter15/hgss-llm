@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=.\.venv\Scripts\python.exe"
if not defined RUN_DIR set "RUN_DIR=.\hgss100m_run"
if not defined TOKENIZER set "TOKENIZER=.\crystal10b\tokenizer.json"
if not defined PORT set "PORT=7860"

"%PYTHON%" -c "import gradio, tokenizers" >nul 2>&1
if errorlevel 1 (
  echo Installing Gradio UI dependencies...
  "%PYTHON%" -m pip install -U gradio tokenizers
  if errorlevel 1 goto :error
)

set "CHECKPOINT_ARG="
if defined CHECKPOINT set "CHECKPOINT_ARG=--checkpoint %CHECKPOINT%"

echo Starting HGSS UI at http://127.0.0.1:%PORT%
"%PYTHON%" "hgss_serve.py" --run-dir "%RUN_DIR%" --tokenizer "%TOKENIZER%" --port %PORT% %CHECKPOINT_ARG%
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo HGSS UI stopped because of an error.
pause
exit /b 1
