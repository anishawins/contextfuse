#!/usr/bin/env bash
# Run this ONCE, from the contextfuse folder, in your normal macOS Terminal.
set -euo pipefail

echo "==> python version"
python3 --version

echo "==> creating virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip --quiet

echo "==> installing requirements (this takes a few minutes)"
pip install -r requirements.txt

echo "==> freezing exact versions actually installed"
pip freeze > requirements.lock.txt

echo "==> checking Apple GPU (MPS) availability"
python3 - <<'PY'
import torch, platform
print("machine        :", platform.machine())
print("torch          :", torch.__version__)
print("MPS available  :", torch.backends.mps.is_available())
print("MPS built      :", torch.backends.mps.is_built())
PY

echo
echo "DONE. Paste the output above back to Claude."
