"""WSGI entrypoint for production servers such as Waitress."""

from app import create_initialized_app
from app.config import Config


app = create_initialized_app(model_path=Config.MODEL_PATH, voices_dir=Config.VOICES_DIR)
