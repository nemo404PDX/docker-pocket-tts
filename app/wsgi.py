"""WSGI entrypoint for production servers such as Waitress."""

from app import create_initialized_app
from app.config import Config


# Forward the config values that server.py would pass via CLI overrides so the
# container path behaves identically (notably skip_voice_normalize — without it,
# every start rewrites voice WAVs in-place on the bind-mounted volume).
app = create_initialized_app(
    model_path=Config.MODEL_PATH,
    voices_dir=Config.VOICES_DIR,
    skip_voice_normalize=Config.SKIP_VOICE_NORMALIZE,
)
