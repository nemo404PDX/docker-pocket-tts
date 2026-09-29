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

# Use the project venv's `hf` CLI when present (POSIX or Windows layout), else a globally-
# installed `hf` on PATH. (huggingface_hub has no __main__, so we call its console script.)
HF_CLI=""
for HF_BIN in \
  "$SERVICE_ROOT/.venv/bin/hf" \
  "$SERVICE_ROOT/.venv/Scripts/hf.exe"; do
  if [ -x "$HF_BIN" ]; then
    HF_CLI="$HF_BIN"
    break
  fi
done
if [ -z "$HF_CLI" ] && command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
fi
if [ -z "$HF_CLI" ]; then
  echo "Error: no 'hf' CLI found (install huggingface_hub into the venv, or put 'hf' on PATH)." >&2
  exit 1
fi

USER_TOKEN="${1:-$HF_TOKEN}"
if [ -z "$USER_TOKEN" ]; then
  echo "USER_TOKEN not provided (pass it as \$1 or set HF_TOKEN in .env)"
  exit 1
fi

# The app resolves weights via hf_hub_download(), which reads the Hugging Face *cache* layout
# under HUGGINGFACE_HUB_CACHE (= POCKET_TTS_MODEL_CACHE_DIR). So we must populate that cache
# dir — NOT a flat --local-dir, which the app never looks at. Default matches Config.MODEL_CACHE_DIR.
CACHE_DIR="${POCKET_TTS_MODEL_CACHE_DIR:-$SERVICE_ROOT/data/models/pocket-tts}"

# The default English config pulls from two repos: model weights from kyutai/pocket-tts and
# the tokenizer from kyutai/pocket-tts-without-voice-cloning. Download both so offline load
# works without a second network hit. Override with $2 / MODEL_ID to fetch only one repo.
REPO="${2:-${MODEL_ID:-kyutai/pocket-tts}}"
SECOND_REPO="kyutai/pocket-tts-without-voice-cloning"

export HF_TOKEN="$USER_TOKEN"
# Point the cache at our dir so this download and later hf_hub_download share one location.
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
mkdir -p "$CACHE_DIR"

echo "Downloading '$REPO' into HF cache: $CACHE_DIR"
$HF_CLI download "$REPO"
if [ "$REPO" != "$SECOND_REPO" ]; then
  echo "Downloading '$SECOND_REPO' (tokenizer) into HF cache: $CACHE_DIR"
  $HF_CLI download "$SECOND_REPO"
fi

echo "Done. The server will now load from this cache without hitting HuggingFace."
