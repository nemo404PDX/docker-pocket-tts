# docker-pocket-tts

A self-hosted, OpenAI-compatible TTS server built on [Kyutai Pocket TTS](https://kyutai-labs.github.io/pocket-tts/). Runs on CPU, streams audio, and accepts custom voice files. Designed to be deployed as a Docker container with data (models / voices) mounted from the host.

- **API:** OpenAI-compatible `POST /v1/audio/speech` (plus `/tts`, `/audio` aliases), streaming support
- **Runtime:** Flask + Waitress, CPU-only PyTorch
- **Voices:** ~27 built-in Pocket TTS voices plus any `.wav`/`.mp3`/`.flac` you drop in `data/voices/`

---

## Quick start (Docker)

```bash
# 1. Configure
cp .env.example .env          # then edit values as needed

# 2. Put your HF token where the container expects it (optional, for voice cloning)
#    The compose file mounts ${HF_TOKEN_FILE:-C:\Users\nemo\.secrets\hf_token} at /run/secrets/hf_token.
#    Override the host path by exporting HF_TOKEN_FILE=<path> before running compose.

# 3. Run
cd deploy
docker compose up --build -d
```

The service listens on `http://localhost:49112`. Health check: `GET /health`, readiness (model loaded): `GET /ready`.

### Try it

```bash
curl http://localhost:49112/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from Pocket TTS.", "voice": "alba", "response_format": "mp3"}' \
  --output out.mp3

# Streaming (chunked audio)
curl -N http://localhost:49112/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Streaming works.", "stream": true}'
```

---

## Local development

No Docker needed. A virtualenv is enough (the repo uses `uv` to create it, but plain `python -m venv` works too).

```bash
# Option A: the helper script (installs CPU deps + pytest into .venv)
scripts/bash/install_pocket_tts.sh

# Option B: manually
uv venv --python 3.11 .venv
.venv/Scripts/python -m pip install -r requirements/cpu_requirements.txt -r requirements/dev_requirements.txt   # Windows
# .venv/bin/python -m pip install ...                                                                          # Linux/macOS

# Run the server directly (reads .env automatically)
python server.py --port 49112

# Run the tests
.venv/Scripts/python -m pytest tests/ -q
```

`requirements/cpu_requirements.txt` pins a CPU-only PyTorch index so you never pull multi-GB CUDA wheels. `pocket-tts` is pinned to an exact version; bump it deliberately (the test suite and voice list track the installed package).

---

## API reference

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/v1/audio/speech`, `/tts`, `/audio` | Generate speech (OpenAI-compatible) |
| `GET`  | `/v1/voices` | List available voices (built-in + custom) |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready` | Readiness — model loaded? |
| `POST` | `/debug/normalize` | Show preprocessing output. **Disabled unless `POCKET_TTS_DEBUG_LOG=true`.** |
| `GET`  | `/` | JSON service stub |

### Speech request body

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `input` (or `text`) | string | — | **Required.** Text to synthesize. |
| `voice` | string | `POCKET_TTS_DEFAULT_VOICE` | Built-in name, a file in the voices dir, or an absolute path. Unknown IDs fall back to the default when `POCKET_TTS_FALLBACK_UNKNOWN_VOICE=true`. |
| `stream` | bool | `POCKET_TTS_STREAM_DEFAULT` | Stream audio chunks instead of one response. |
| `response_format` | string | `mp3` | One of `mp3`, `wav`, `opus`, `aac`, `flac`, `pcm` (`mpeg` → `mp3`; unknown values fall back to `wav`). |

Per-request overrides for the pipeline are also accepted: `normalize`, `segmentation` (`none`/`paragraph`/`sentence`), `skip_code_blocks`, `code_block_phrase`, `emoji_handling` (`strip`/`describe`/`keep`).

---

## Configuration (environment variables)

All settings come from environment variables with sensible defaults. In Docker, `deploy/docker-compose.yml` sets the **container-internal** paths and reads everything else from `.env`. For local runs, `server.py` loads `.env` automatically.

### Server
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_HOST` | `0.0.0.0` | Bind host (local runs; the Docker CMD binds `0.0.0.0`). |
| `POCKET_TTS_PORT` | `49112` | Port. Must match the left side of the compose port mapping. |
| `POCKET_TTS_THREADS` | `4` | Waitress worker threads (only used by `python server.py`, not the container path). |

### Request hardening & security
All optional — defaults preserve existing behavior, so nothing changes unless you set them.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_MAX_INPUT_CHARS` | `32000` | Reject synthesis input longer than this (HTTP 413). Set `0` to disable the cap. |
| `POCKET_TTS_API_KEY` | — (unset) | When set, every route except `/health` and `/ready` requires `Authorization: Bearer <key>`. Leave unset for open local use. |
| `POCKET_TTS_SEGMENT_GAP_MS` | `200` | Silence inserted between synthesized segments so long text doesn't run together (set `0` to disable). |
| `POCKET_TTS_MAX_CACHED_VOICES` | `8` | Bound for the loaded-voice LRU cache (set `0` for unbounded). |

### Model
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_MODEL_PATH` | — | Explicit model config path (`.yaml`) or variant name; overrides the default. |
| `POCKET_TTS_MODEL_CACHE_DIR` | `<data>/models/pocket-tts` | HF cache dir. Compose sets `/data/models/pocket-tts`. Pre-populate it (`hf download kyutai/pocket-tts --local-dir …`) and the server loads weights from disk without hitting HuggingFace at startup. |
| `POCKET_TTS_REQUIRE_VOICE_CLONING` | `false` | Fail startup if voice-cloning weights are missing (else fall back to without-voice-cloning). |

### Voices
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_VOICES_DIR` | `<data>/voices` if it exists, else none | Directory of custom voice files. Compose sets `/data/voices`. |
| `POCKET_TTS_DEFAULT_VOICE` | `default.wav` (if present) else `alba` | Default voice for requests that omit one. |
| `POCKET_TTS_FALLBACK_UNKNOWN_VOICE` | `true` | Fall back to the default instead of 400 on unknown voice IDs. |
| `POCKET_TTS_SKIP_VOICE_NORMALIZE` | `false` | Skip startup loudness normalization of custom WAVs. |

### Text pipeline
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_BRIEFING_MODE` | `false` | Long-transcript profile: forces normalize + sentence segmentation and enables source-link stripping (unless overridden). |
| `POCKET_TTS_NORMALIZE_DEFAULT` | `false` | Normalize text by default. |
| `POCKET_TTS_SEGMENTATION_DEFAULT` | `none` | Default segmentation mode (`none`/`paragraph`/`sentence`). |
| `POCKET_TTS_DROP_SOURCE_LINKS` | follows briefing mode | Strip a trailing `--- SOURCE LINKS ---` section. |
| `POCKET_TTS_SKIP_CODE_BLOCKS` | `true` | Replace fenced code blocks with a placeholder before synthesis. |
| `POCKET_TTS_CODE_BLOCK_PHRASE` | `, see the code in the conversation history,` | Placeholder text for skipped code blocks. |
| `POCKET_TTS_EMOJI_HANDLING` | `strip` | `strip`, `describe`, or `keep`. |

### Logging
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_LOG_LEVEL` | `INFO` | Log level. A later explicit `--log-level` (server.py) overrides this. |
| `POCKET_TTS_LOG_TO_FILE` | `false` | Also write a rotating file at `$POCKET_TTS_LOG_DIR/pocket_tts.log`. Under Docker, stdout is always captured by the container log driver (rotated in compose), so leave this off unless you add a `/data/logs` volume. |
| `POCKET_TTS_LOG_DIR` | `<data>/logs` | Log directory for file logging. |

### Hugging Face token
The app reads `HF_TOKEN`, or falls back to the file at `HF_TOKEN_FILE` (default `/run/secrets/hf_token`). The compose file mounts a host file there and sets `HF_TOKEN_FILE` accordingly; override the **host** path with `HF_TOKEN_FILE=<path>` in your shell env.

---

## Voices

- **Built-in:** ~27 Pocket TTS voices (`alba`, `marius`, `jean`, …). The list is derived from the installed `pocket-tts` package, so it tracks upstream automatically — see `GET /v1/voices`.
- **Custom:** drop `.wav`/`.mp3`/`.flac` files into `data/voices/`. Reference them by filename (e.g. `"voice": "ada.wav"`) or bare stem if unique. Startup normalizes WAV loudness to a target level (disable with `POCKET_TTS_SKIP_VOICE_NORMALIZE`).

Shipped custom voices (short names, easy to thumb-type in a GUI / mobile client):

| File | Voice id (bare) |
| ---- | --------------- |
| `ada.wav`   | `ada` |
| `amy.wav`   | `amy` |
| `eli.wav`   | `eli` |
| `ellie.wav` | `ellie` |
| `emi.wav`   | `emi` |
| `max.wav`   | `max` |
| `default.wav` | `default` (also the default voice when present) |

> Note: a custom file whose name matches a built-in voice is shadowed — the built-in wins for bare-name requests; use the full filename to force the local file. The shipped names above were chosen specifically so none collide with a built-in, so every one resolves by its short id.

---

## Logging flow

```
setup_logging()  →  console handler → stdout   ALWAYS on
                          └─ in Docker: captured by the container log driver (json-file, 10 MB × 5)
                       file handler → $LOG_DIR/pocket_tts.log   only if POCKET_TTS_LOG_TO_FILE=true
```

There is no separate access log; per-request lines come from the app's own TTS input/output logging.

---

## Tests & CI

`tests/` covers audio conversion and text normalization. Run locally with `pytest tests/ -q`. GitHub Actions (`.github/workflows/tests.yml`) installs the CPU requirements + pytest on Python 3.11 and runs the suite on every push to `master`/`main` and on PRs.
