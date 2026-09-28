# AGENTS.md — working notes for AI agents / contributors

Conventions and gotchas specific to this repo. Read before making changes.

## Layout
- `app/` — Flask app. `config.py` (env → Config), `logging_config.py`, `api/routes.py` (blueprint), `services/{tts,audio,normalize,preprocess}.py`.
- `server.py` — local dev entrypoint (argparse + Waitress). Loads `.env` itself.
- `app/wsgi.py` — container entrypoint (`waitress-serve app.wsgi:app`). **Passes no CLI overrides** — anything that must work in Docker has to flow through `Config` → `create_app()`, not just `server.py`.
- `deploy/` — `Dockerfile`, `docker-compose.yml`, `entrypoint.sh`.
- `requirements/cpu_requirements.txt` (runtime, CPU torch) + `dev_requirements.txt` (pytest).
- `tests/` — pytest suite.

## Hard rules
1. **Keep the Docker path working.** The container runs via `wsgi.py`, not `server.py`. If you add a config knob or behavior, make sure it reaches `app.config` from `Config` (see how `TTS_BRIEFING_MODE`/`TTS_DROP_SOURCE_LINKS` are propagated in `create_app`).
2. **Don't hardcode what the installed package already knows.** Built-in voice names come from `_builtin_voice_names()` in `config.py`, which reads pocket-tts' own mapping. Don't re-hardcode a list that will drift.
3. **Pin deliberately.** `pocket-tts` is pinned to an exact version and torch/torchaudio are capped at `<=2.8`. Bumping them can break the test suite or pull CUDA wheels — do it on purpose, then run tests.
4. **CPU-only.** Always install via `requirements/cpu_requirements.txt` (it sets the CPU PyTorch index). Never add a bare `pip install torch`.
5. **Secrets stay out of git.** `.env` is gitignored and untracked; only `.env.example` is committed. The HF token reaches the container via a mounted file (`HF_TOKEN_FILE`), not an env var in compose.

## Testing
```bash
.venv/Scripts/python -m pytest tests/ -q     # Windows
# .venv/bin/python -m pytest tests/ -q       # Linux/macOS
```
CI (`.github/workflows/tests.yml`) runs the same on Python 3.11 with CPU requirements. Keep it green; it's the only automated guardrail.

## Gotchas
- **Log-level precedence:** `setup_logging(level)` re-applies an *explicit* level to existing handlers, but a no-arg call leaves the first caller authoritative. Don't rely on calling it twice with defaults to change the level.
- **`/debug/normalize`** is gated behind `POCKET_TTS_DEBUG_LOG=true` (it echoes request content). Leave it off in production.
- **Voice shadowing:** a custom file named like a built-in (`alba.wav`) loses to the built-in for bare-name requests; clients must send the full filename to force the local file.
- **Compose vs `.env`:** compose's `environment:` block only sets container-internal paths + `HF_TOKEN_FILE`. Everything user-facing lives in `.env`/`.env.example`, not compose.
