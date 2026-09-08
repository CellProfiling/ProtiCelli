#!/usr/bin/env sh
set -eu

PROTICELLI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROTICELLI_ENV="$PROTICELLI_ROOT/.venv"
PROTICELLI_PYTHON="$PROTICELLI_ENV/bin/python"
PROTICELLI_OS=$(uname -s 2>/dev/null || printf 'Unknown')

cd "$PROTICELLI_ROOT"

proticelli_find_compatible_python() {
  for PROTICELLI_CANDIDATE in \
    python3.13 python3.12 python3.11 python3.10 python3 python \
    /opt/homebrew/bin/python3 /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
    /usr/bin/python3
  do
    if command -v "$PROTICELLI_CANDIDATE" >/dev/null 2>&1; then
      PROTICELLI_CANDIDATE_PATH=$(command -v "$PROTICELLI_CANDIDATE")
      if "$PROTICELLI_CANDIDATE_PATH" -c \
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1
      then
        printf '%s\n' "$PROTICELLI_CANDIDATE_PATH"
        return 0
      fi
    fi
  done
  return 1
}

proticelli_remove_incomplete_environment() {
  # This exact path is private to the Gallery and may contain a partially
  # created venv/Conda prefix after a failed bootstrap attempt.
  if [ -e "$PROTICELLI_ENV" ]; then
    rm -rf -- "$PROTICELLI_ENV"
  fi
}

proticelli_show_detected_python() {
  for PROTICELLI_CANDIDATE in python3 python
  do
    if command -v "$PROTICELLI_CANDIDATE" >/dev/null 2>&1; then
      PROTICELLI_CANDIDATE_PATH=$(command -v "$PROTICELLI_CANDIDATE")
      PROTICELLI_CANDIDATE_VERSION=$(
        "$PROTICELLI_CANDIDATE_PATH" --version 2>&1 || printf 'version unavailable'
      )
      echo "  $PROTICELLI_CANDIDATE_PATH - $PROTICELLI_CANDIDATE_VERSION"
    fi
  done
}

: "${PROTICELLI_WEB_DEVICE:=auto}"
export PROTICELLI_WEB_DEVICE
if [ "$PROTICELLI_OS" = "Darwin" ]; then
  : "${PYTORCH_ENABLE_MPS_FALLBACK:=1}"
  export PYTORCH_ENABLE_MPS_FALLBACK
fi

if [ ! -x "$PROTICELLI_PYTHON" ]; then
  if PROTICELLI_BOOTSTRAP=$(proticelli_find_compatible_python); then
    :
  else
    PROTICELLI_BOOTSTRAP=""
  fi

  echo ""
  echo "ProtiCelli Interactive Gallery - first-time setup on $PROTICELLI_OS"
  echo "Creating a private Python 3.10+ environment. This can take several minutes."
  echo ""

  PROTICELLI_ENV_READY=0
  if [ -n "$PROTICELLI_BOOTSTRAP" ]; then
    PROTICELLI_BOOTSTRAP_VERSION=$(
      "$PROTICELLI_BOOTSTRAP" --version 2>&1 || printf 'Python 3.10+'
    )
    echo "Using $PROTICELLI_BOOTSTRAP_VERSION from $PROTICELLI_BOOTSTRAP."
    if "$PROTICELLI_BOOTSTRAP" -m venv "$PROTICELLI_ENV"; then
      PROTICELLI_ENV_READY=1
    else
      echo "The detected Python could not create a virtual environment."
      proticelli_remove_incomplete_environment
    fi
  fi

  if [ "$PROTICELLI_ENV_READY" -eq 0 ]; then
    PROTICELLI_CONDA="${CONDA_EXE:-}"
    if [ -z "$PROTICELLI_CONDA" ] || [ ! -x "$PROTICELLI_CONDA" ]; then
      if command -v conda >/dev/null 2>&1; then
        PROTICELLI_CONDA=$(command -v conda)
      else
        PROTICELLI_CONDA=""
      fi
    fi
    if [ -n "$PROTICELLI_CONDA" ]; then
      echo "No compatible standalone Python was found; creating Python 3.12 with Conda."
      if "$PROTICELLI_CONDA" create --yes --prefix "$PROTICELLI_ENV" python=3.12 pip; then
        PROTICELLI_ENV_READY=1
      else
        echo "Conda could not create the private environment; trying another available method."
        proticelli_remove_incomplete_environment
      fi
    fi
  fi

  if [ "$PROTICELLI_ENV_READY" -eq 0 ] && command -v uv >/dev/null 2>&1; then
    PROTICELLI_UV=$(command -v uv)
    echo "No compatible Python was found; installing private Python 3.12 with uv."
    if "$PROTICELLI_UV" python install 3.12 && \
      "$PROTICELLI_UV" venv --python 3.12 "$PROTICELLI_ENV"
    then
      PROTICELLI_ENV_READY=1
    else
      echo "uv could not create the private environment."
      proticelli_remove_incomplete_environment
    fi
  fi

  if [ "$PROTICELLI_ENV_READY" -eq 0 ]; then
    echo ""
    echo "ProtiCelli could not find or automatically install Python 3.9 or newer."
    echo "Detected commands:"
    proticelli_show_detected_python
    echo ""
    echo "Install Python 3.12, Conda, or uv, then run this launcher again."
    echo "Python: https://www.python.org/downloads/macos/"
    exit 1
  fi

  if ! "$PROTICELLI_PYTHON" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    >/dev/null 2>&1
  then
    echo "The private environment was created, but it does not contain Python 3.10 or newer."
    exit 1
  fi

  "$PROTICELLI_PYTHON" -m pip install --upgrade pip
  "$PROTICELLI_PYTHON" -m pip install -e ".[web]"
fi

if [ "${PROTICELLI_SKIP_ASSET_DOWNLOAD:-0}" != "1" ]; then
  if ! "$PROTICELLI_PYTHON" -m proticelli.utils.download --check >/dev/null 2>&1; then
    echo ""
    echo "Downloading ProtiCelli model assets. This happens once and may take several minutes."
    echo "Keep this window open; interrupted downloads are retried safely on the next launch."
    echo ""
    if ! "$PROTICELLI_PYTHON" -m proticelli.utils.download; then
      echo "Model asset download failed. Check the internet connection and run this launcher again."
      echo "Existing complete assets were not changed."
      exit 1
    fi
  fi
fi

if [ "$PROTICELLI_OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$PROTICELLI_PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo ""
    echo "NVIDIA GPU detected, but PyTorch cannot use it."
    echo "Run ./proticelli-enable-nvidia.sh once, then restart the Gallery."
    echo ""
  fi
elif [ "$PROTICELLI_OS" = "Darwin" ]; then
  if "$PROTICELLI_PYTHON" -c 'import platform; raise SystemExit(0 if platform.machine() == "arm64" else 1)' >/dev/null 2>&1; then
    if ! "$PROTICELLI_PYTHON" -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)' >/dev/null 2>&1; then
      echo ""
      echo "Apple Silicon detected, but PyTorch MPS is unavailable; ProtiCelli will use CPU."
      echo "Run ./proticelli-local.sh --diagnose for details."
      echo ""
    fi
  fi
fi

exec "$PROTICELLI_PYTHON" -m proticelli_web --mode auto "$@"
