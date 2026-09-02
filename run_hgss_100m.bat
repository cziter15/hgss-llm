@echo off
setlocal EnableExtensions

REM Fast full-quaternion-read experiment: 99.42M parameters / 100M tokens.
REM Expected runtime on RTX 5060 Ti 16 GB: benchmark before relying on estimate.

if not defined OUT_DIR set "OUT_DIR=.\hgss100m_run"
if not defined TOKENS set "TOKENS=100000000"

REM 99,415,168 parameters with full_quaternion_read=True.
if not defined DIM set "DIM=800"
if not defined LAYERS set "LAYERS=12"
if not defined HEADS set "HEADS=8"
if not defined D_K set "D_K=32"
if not defined D_V set "D_V=64"
if not defined D_CONV set "D_CONV=4"

REM Effective batch: 4 * 2048 * 8 = 65,536 tokens; ~1,526 steps.
if not defined MICRO_BATCH set "MICRO_BATCH=4"
if not defined GRAD_ACCUM set "GRAD_ACCUM=8"

REM This model fits without activation recomputation.
if not defined CHECKPOINT_STRIDE set "CHECKPOINT_STRIDE=0"
if not defined LOG_EVERY set "LOG_EVERY=1"
if not defined SAVE_EVERY set "SAVE_EVERY=100"

if not defined LR set "LR=3e-4"
if not defined MIN_LR set "MIN_LR=3e-5"
if not defined WARMUP_STEPS set "WARMUP_STEPS=100"

call "%~dp0run_hgss_1b.bat"
exit /b %errorlevel%
