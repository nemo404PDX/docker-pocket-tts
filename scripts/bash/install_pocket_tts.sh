#!/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${SERVICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [ -f "$SERVICE_ROOT/.venv/bin/activate" ]; then
  source "$SERVICE_ROOT/.venv/bin/activate"
else
  echo ".venv/bin/activate not found at $SERVICE_ROOT/.venv/bin/activate"
  exit 1
fi

echo " "
echo Installing /kyutai Pocket TTS...
echo " "

pip install --upgrade pip setuptools wheel

pip install pocket-tts
