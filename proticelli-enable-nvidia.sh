#!/usr/bin/env sh
set -eu

PROTICELLI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROTICELLI_PYTHON="$PROTICELLI_ROOT/.venv/bin/python"
PROTICELLI_TORCH_INDEX_URL=${PROTICELLI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}

cd "$PROTICELLI_ROOT"

if [ "$(uname -s 2>/dev/null || printf 'Unknown')" != "Linux" ]; then
  echo "This helper is for Linux with an NVIDIA GPU."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver was detected. Install or update the NVIDIA driver first."
  exit 1
fi

if [ ! -x "$PROTICELLI_PYTHON" ]; then
  echo "No ProtiCelli environment was found. Run ./proticelli-local.sh first."
  exit 1
fi

echo ""
echo "ProtiCelli NVIDIA acceleration setup for Linux"
echo "This installs the PyTorch CUDA wheel into the Gallery's private .venv."
echo "Wheel source: $PROTICELLI_TORCH_INDEX_URL"
echo "The download is large and can take several minutes."
printf "Continue? [y/N] "
read -r PROTICELLI_REPLY
case "$PROTICELLI_REPLY" in
  y|Y|yes|YES) ;;
  *) exit 0 ;;
esac

"$PROTICELLI_PYTHON" -m pip install --upgrade "torch>=2.0,<2.9" "torchvision<0.24" --index-url "$PROTICELLI_TORCH_INDEX_URL"
"$PROTICELLI_PYTHON" -c "import torch,sys; print('PyTorch:', torch.__version__); print('CUDA/ROCm build:', torch.version.cuda or getattr(torch.version, 'hip', None)); print('Accelerator available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); sys.exit(0 if torch.cuda.is_available() else 2)"

echo ""
echo "GPU acceleration is ready. Restart ProtiCelli Interactive Gallery."
