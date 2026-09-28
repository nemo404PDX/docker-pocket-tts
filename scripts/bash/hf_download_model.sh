#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="${SERVICE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

ENV_FILE="$SERVICE_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo ".env not found at $ENV_FILE"
  exit 1
fi

if [ -f "$SERVICE_ROOT/.venv/bin/activate" ]; then
  source "$SERVICE_ROOT/.venv/bin/activate"
fi

USER_TOKEN="${1:-$HF_TOKEN}"
if [ -z "$USER_TOKEN" ]; then
  echo "USER_TOKEN not provided"
  exit 1
fi

# Default to the model the app actually loads (see app/services/tts.py)
MODEL_ID="${2:-${MODEL_ID:-kyutai/pocket-tts}}"
if [ -z "$MODEL_ID" ]; then
  echo "MODEL_ID not provided"
  exit 1
fi

export HF_TOKEN=$USER_TOKEN
cd "$SERVICE_ROOT"
hf download "$MODEL_ID" --local-dir "./data/models/$MODEL_ID"
