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
| `GET`  | `/v1/voices` | List available voices (built-in + custom). Each entry includes a stable `created_at` (epoch seconds; built-ins are `0`, custom voices carry their file mtime) for OpenAI compatibility. |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready` | Readiness — model loaded? |
| `POST` | `/debug/normalize` | Show preprocessing output. **Disabled unless `POCKET_TTS_DEBUG_LOG=true`.** |
| `GET`  | `/` | JSON service info (includes `supported_formats`) |

### Speech request body

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `input` (or `text`) | string | — | **Required.** Text to synthesize. Capped at `POCKET_TTS_MAX_INPUT_CHARS` (default 32000) → HTTP 413 if exceeded. |
| `voice` | string | `POCKET_TTS_DEFAULT_VOICE` | Built-in name, a file in the voices dir, or an absolute path. Unknown IDs fall back to the default when `POCKET_TTS_FALLBACK_UNKNOWN_VOICE=true`. |
| `stream` | bool | `POCKET_TTS_STREAM_DEFAULT` | Stream audio chunks instead of one response (PCM/WAV only; other formats return a full file). |
| `response_format` | string | `mp3` | One of the formats this environment can encode — see `GET /` → `supported_formats`. Always: `wav`, `mp3`, `flac`, `pcm`; plus `aac`/`opus` only if an ffmpeg *audio backend* is registered with torchaudio (the CPU build ships soundfile-only, so they're unavailable in both local dev and the Docker image today). `mpeg` → `mp3`. Unsupported values return **400** with the supported list. |

Errors use the OpenAI-shaped envelope: `{"error": {"message", "type", "code"}}` (e.g. 413 `input_too_long`, 400 `unsupported_format` / `unknown_voice`, 500 `generation_failed`).

Per-request overrides for the pipeline are also accepted: `normalize`, `segmentation` (`none`/`paragraph`/`sentence`), `skip_code_blocks`, `code_block_phrase`, `emoji_handling` (`strip`/`describe`/`keep`), and `marker_pattern` (regex of client markers to strip; empty value disables stripping).

**Text normalization & pronunciation.** When `normalize` is on, the pipeline expands contractions and abbreviations, strips markdown/code blocks, and — so every voice pronounces tech terms consistently — rewrites all-caps acronyms into their spoken form before synthesis: letter-initialisms are spaced out (`IP` → "I P", `CPU` → "C P U", `HTTPS` → "H T T P S") and common ones become phrases (`LLM` → "large language model", `TTS` → "text to speech"). Matching is case-sensitive on the all-caps token, so mixed-case words like `ffmpeg` or `iPhone` are left as-is. The list lives in `app/services/normalize.py` (`ACRONYM_EXPANSIONS`) — add any term you hear mangled there.

**Cut-off at sentence ends.** pocket-tts stops generating a short tail after it detects end-of-speech; the default guess (~3 frames ≈ 240 ms) is sometimes too short, so a voice can clip its final consonant/syllable (e.g. "…sharp" → "…shar"). This is stochastic and varies per voice file, which is why one reference audio may say "IP" fine while another doesn't. `POCKET_TTS_FRAMES_AFTER_EOS` lets you lengthen that tail to reduce clipping; because sampling is non-deterministic it's a tuning lever, not a guarantee — if a specific voice still clips, also check its reference WAV for trailing noise (see Voices below).

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
| `POCKET_TTS_SEGMENT_GAP_MS` | `200` | Silence inserted between segments at **sentence** boundaries (a segment ending in `.`, `!`, or `?`) so long text doesn't run together. Set `0` to disable all inter-segment silence. |
| `POCKET_TTS_SEGMENT_SEAM_MS` | `30` | Silence inserted between **mid-sentence** fragments (where a single sentence was split by the length/token caps). Kept small so long clauses flow as one phrase instead of getting an audible pause in the middle. Capped at `SEGMENT_GAP_MS`; set `0` for no seam. |
| `POCKET_TTS_MAX_CACHED_VOICES` | `8` | Bound for the loaded-voice LRU cache (set `0` for unbounded). |
| `POCKET_TTS_FRAMES_AFTER_EOS` | — (unset) | Extra decoder frames after end-of-speech (1 frame = 80 ms). Unset uses pocket-tts's built-in guess (~3 frames), which is short enough that some voices clip the final consonant/syllable. Set e.g. `6` for a longer tail; `0` disables it. Generation is stochastic, so treat this as a per-voice tuning lever rather than a guaranteed fix. |

### Model
| Variable | Default | Description |
| -------- | ------- | ----------- |
| `POCKET_TTS_MODEL_PATH` | — | Explicit model config path (`.yaml`) or variant name; overrides the default. |
| `POCKET_TTS_MODEL_CACHE_DIR` | `<repo>/data/models/pocket-tts` | HF cache dir the server reads weights from (compose sets `/data/models/pocket-tts`). Pre-populate it with `scripts/bash/hf_download_model.sh <token>` — it downloads both repos into this cache so the server loads offline without hitting HuggingFace at startup. |
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
| `POCKET_TTS_HERMES_MARKER_PATTERN` | Hermes markers (`[terminal]`, `[OUT-OF-BAND USER MESSAGE]`) | Regex of client markers to strip (whole line removed). Override if you feed a different client's output; set empty to disable. Also overridable per-request via `marker_pattern`. |

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
- **Custom:** drop `.wav`/`.mp3`/`.flac` files into `data/voices/`. Reference them by filename (e.g. `"voice": "ada.wav"`) or bare stem if unique. Startup normalizes WAV loudness to a target level (disable with `POCKET_TTS_SKIP_VOICE_NORMALIZE`). The reference audio is used for zero-shot voice cloning, so its quality directly shapes the output — background noise in the source (e.g. pulled from a video) can come through as a faint "rush" or hum, and a noisy/short clip also makes end-of-speech detection less reliable. For cleanest results use a dry, well-recorded sample with no music or room tone.

Shipped custom voices (short names, easy to thumb-type in a GUI / mobile client):

| File | Voice id (bare) |
| ---- | --------------- |
| `ada.wav`     | `ada` |
| `amy.wav`     | `amy` |
| `eli.wav`     | `eli` |
| `emi.wav`     | `emi` |
| `liz.wav`     | `liz` |
| `max.wav`     | `max` |
| `may.wav`     | `may` |
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
