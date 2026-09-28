#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${SERVICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Locate the project venv python (POSIX or Windows layout)
if [ -x "$SERVICE_ROOT/.venv/bin/python" ]; then
  VENV_PY="$SERVICE_ROOT/.venv/bin/python"
elif [ -x "$SERVICE_ROOT/.venv/Scripts/python.exe" ]; then
  VENV_PY="$SERVICE_ROOT/.venv/Scripts/python.exe"
else
  echo "Error: no .venv found at $SERVICE_ROOT/.venv" >&2
  echo "Create one first, e.g.: uv venv --python 3.11 \"$SERVICE_ROOT/.venv\"" >&2
  exit 1
fi

echo "Installing Pocket TTS + project dependencies into:"
echo "  $VENV_PY"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
# cpu_requirements.txt includes the CPU-only torch index (matches the Docker image),
# so this never pulls multi-GB CUDA wheels. dev_requirements adds pytest for tests/.
"$VENV_PY" -m pip install \
  -r "$SERVICE_ROOT/requirements/cpu_requirements.txt" \
  -r "$SERVICE_ROOT/requirements/dev_requirements.txt"

echo "Done."
