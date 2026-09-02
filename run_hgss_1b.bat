@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM HGSS ~1.198B full-quaternion-read / RTX 5060 Ti 16 GB / 10B token pretraining
REM Windows .bat version
REM ============================================================
REM
REM Expected files in this directory:
REM   hgss_1b_train.py
REM   model(20260902-145037).py
REM   triton_scan(5).py
REM
REM Dataset:
REM   IFM/CrystalCoderDatasets
REM   already tokenized + packed
REM   70%% SlimPajama text / 30%% StarCoder FIM
REM   sequence length = 2048
REM
REM Default model:
REM   dim=2048
REM   layers=18
REM   heads=16
REM   d_k=64
REM   d_v=128
REM   ~1.198B parameters with full quaternion read
REM
REM HGSS memory horizons:
REM   memory_min = 4
REM   memory_max = 131072 (128k)
REM
REM Usage:
REM   run_hgss_1b.bat
REM By default training resumes from OUT_DIR\latest.txt when it exists.
REM Set FRESH_START=1 to ignore it, or RESUME to an explicit checkpoint path.
REM
REM Quick 100M test:
REM   set TOKENS=100000000
REM   set DATA_DIR=.\crystal_test
REM   call run_hgss_1b.bat
REM ============================================================

cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"

REM ------------------------------------------------------------
REM Defaults - may be overridden before calling this .bat
REM ------------------------------------------------------------

if not defined PYTHON set "PYTHON=.\.venv\Scripts\python.exe"

if not defined DATA_DIR set "DATA_DIR=.\crystal10b"
if not defined OUT_DIR set "OUT_DIR=.\hgss1b_run"

set "RESUME_ARGS="
if defined RESUME set RESUME_ARGS=--resume "!RESUME!"
if "%FRESH_START%"=="1" set "RESUME_ARGS=--no-auto-resume"

if not defined MODEL_FILE set "MODEL_FILE=.\hgss\model.py"
if not defined TRITON_FILE set "TRITON_FILE=.\hgss\triton_scan.py"

if not defined TOKENS set "TOKENS=10000000000"
if not defined CODE_FRACTION set "CODE_FRACTION=0.30"

if not defined DIM set "DIM=2048"
if not defined LAYERS set "LAYERS=18"
if not defined HEADS set "HEADS=16"
if not defined D_K set "D_K=64"
if not defined D_V set "D_V=128"
if not defined D_CONV set "D_CONV=4"

if not defined SEQ_LEN set "SEQ_LEN=2048"
if not defined MICRO_BATCH set "MICRO_BATCH=1"
if not defined GRAD_ACCUM set "GRAD_ACCUM=64"
if not defined CHECKPOINT_STRIDE set "CHECKPOINT_STRIDE=2"
if not defined LOG_EVERY set "LOG_EVERY=1"
if not defined SAVE_EVERY set "SAVE_EVERY=50"

if "%CHECKPOINT_STRIDE%"=="0" (
  set "CHECKPOINT_ARGS=--no-block-checkpoint"
) else (
  set "CHECKPOINT_ARGS=--checkpoint-stride %CHECKPOINT_STRIDE%"
)

if not defined LR set "LR=2e-4"
if not defined MIN_LR set "MIN_LR=2e-5"
if not defined WARMUP_STEPS set "WARMUP_STEPS=1000"
if not defined WEIGHT_DECAY set "WEIGHT_DECAY=0.1"
if not defined GRAD_CLIP set "GRAD_CLIP=1.0"

if not defined MEMORY_MIN set "MEMORY_MIN=4"
if not defined MEMORY_MAX set "MEMORY_MAX=131072"

echo.
echo ============================================================
echo  HGSS pretraining
echo ============================================================
echo Dataset dir : %DATA_DIR%
echo Output dir  : %OUT_DIR%
if defined RESUME echo Resume      : %RESUME%
if "%FRESH_START%"=="1" echo Resume      : disabled (fresh start)
if not defined RESUME if not "%FRESH_START%"=="1" echo Resume      : automatic from latest.txt
echo Token budget: %TOKENS%
echo Code fraction: %CODE_FRACTION%
echo Model       : D=%DIM% L=%LAYERS% H=%HEADS% K=%D_K% V=%D_V%
echo Sequence    : %SEQ_LEN%
echo Batch       : micro=%MICRO_BATCH% accumulation=%GRAD_ACCUM%
if "%CHECKPOINT_STRIDE%"=="0" echo Checkpoint  : disabled
if not "%CHECKPOINT_STRIDE%"=="0" echo Checkpoint  : every %CHECKPOINT_STRIDE% blocks
echo Memory      : %MEMORY_MIN% .. %MEMORY_MAX%
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Required files
REM ------------------------------------------------------------

if not exist "hgss_1b_train.py" (
    echo ERROR: hgss_1b_train.py not found in:
    echo   %CD%
    goto :error
)

if not exist "%MODEL_FILE%" (
    echo ERROR: model file not found:
    echo   %MODEL_FILE%
    goto :error
)

if not exist "%TRITON_FILE%" (
    echo ERROR: Triton file not found:
    echo   %TRITON_FILE%
    goto :error
)

REM ------------------------------------------------------------
REM Python / CUDA preflight
REM ------------------------------------------------------------

echo [preflight] Checking Python / PyTorch / CUDA...

"%PYTHON%" -c "import torch,sys; print('PyTorch:',torch.__version__); print('torch CUDA runtime:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: CUDA-enabled PyTorch is not available.
    echo Install a recent PyTorch build suitable for RTX 50-series / Blackwell.
    goto :error
)

"%PYTHON%" -c "import torch; print('GPU:',torch.cuda.get_device_name(0)); print('Compute capability:',torch.cuda.get_device_capability(0)); print('BF16 supported:',torch.cuda.is_bf16_supported())"
if errorlevel 1 goto :error

echo.
echo [1/2] Downloading/resuming already-tokenized dataset...
echo.

"%PYTHON%" "hgss_1b_train.py" download ^
  --data-dir "%DATA_DIR%" ^
  --tokens %TOKENS% ^
  --code-fraction %CODE_FRACTION%

if errorlevel 1 (
    echo.
    echo ERROR: dataset download failed.
    goto :error
)

echo.
echo [2/2] Starting HGSS training...
echo.

"%PYTHON%" "hgss_1b_train.py" train ^
  --data-dir "%DATA_DIR%" ^
  --model-file "%MODEL_FILE%" ^
  --triton-file "%TRITON_FILE%" ^
  --out-dir "%OUT_DIR%" ^
  !RESUME_ARGS! ^
  --tokens %TOKENS% ^
  --code-fraction %CODE_FRACTION% ^
  --seq-len %SEQ_LEN% ^
  --vocab-size 32032 ^
  --boundary-token-id 2 ^
  --dim %DIM% ^
  --layers %LAYERS% ^
  --heads %HEADS% ^
  --d-k %D_K% ^
  --d-v %D_V% ^
  --d-conv %D_CONV% ^
  --memory-min %MEMORY_MIN% ^
  --memory-max %MEMORY_MAX% ^
  --scan-chunk-size 0 ^
  --micro-batch %MICRO_BATCH% ^
  --grad-accum %GRAD_ACCUM% ^
  !CHECKPOINT_ARGS! ^
  --lr %LR% ^
  --min-lr %MIN_LR% ^
  --warmup-steps %WARMUP_STEPS% ^
  --beta1 0.9 ^
  --beta2 0.95 ^
  --weight-decay %WEIGHT_DECAY% ^
  --grad-clip %GRAD_CLIP% ^
  --log-every %LOG_EVERY% ^
  --save-every %SAVE_EVERY%

if errorlevel 1 (
    echo.
    echo ERROR: training stopped with an error.
    goto :error
)

echo.
echo ============================================================
echo  Training finished successfully.
echo ============================================================
goto :eof

:error
echo.
echo ============================================================
echo  HGSS launcher stopped because of an error.
echo ============================================================
exit /b 1
