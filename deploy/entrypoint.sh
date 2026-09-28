#!/bin/bash
set -e
if [ -f /run/secrets/hf_token ]; then
    export HF_TOKEN=$(cat /run/secrets/hf_token)
fi
exec "$@"
