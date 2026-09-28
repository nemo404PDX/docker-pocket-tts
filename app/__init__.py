"""
PocketTTS OpenAI-Compatible Server

Flask application factory and initialization.
"""

from typing import TYPE_CHECKING

from app.config import Config
from app.logging_config import get_logger, setup_logging
from app.services.audio import normalize_voice_wavs

if TYPE_CHECKING:
    from flask import Flask


def create_app(config_overrides: dict = None) -> 'Flask':
    """
    Application factory for creating the Flask app.

    Args:
        config_overrides: Optional dictionary of config values to override

    Returns:
        Configured Flask application
    """
    from flask import Flask

    # Setup logging first
    setup_logging()
    logger = get_logger()

    # Create Flask app with correct paths
    app = Flask(__name__)

    # Apply default config
    app.config['STREAM_DEFAULT'] = Config.STREAM_DEFAULT
    app.config['TEXT_PREPROCESS_DEFAULT'] = Config.TEXT_PREPROCESS_DEFAULT
    app.config['TTS_NORMALIZE_DEFAULT'] = Config.TTS_NORMALIZE_DEFAULT
    app.config['TTS_SEGMENTATION_DEFAULT'] = Config.TTS_SEGMENTATION_DEFAULT
    app.config['TTS_SKIP_CODE_BLOCKS'] = Config.TTS_SKIP_CODE_BLOCKS
    app.config['TTS_CODE_BLOCK_PHRASE'] = Config.TTS_CODE_BLOCK_PHRASE
    app.config['TTS_HERMES_MARKER_PATTERN'] = Config.TTS_HERMES_MARKER_PATTERN
    app.config['TTS_EMOJI_HANDLING'] = Config.TTS_EMOJI_HANDLING
    # Briefing mode + source-links stripping must reach app.config too, otherwise
    # routes._build_pipeline_config() can't see them on the WSGI/container path
    # (server.py passes these as overrides, but wsgi.py does not).
    app.config['TTS_BRIEFING_MODE'] = Config.TTS_BRIEFING_MODE
    app.config['TTS_DROP_SOURCE_LINKS'] = Config.TTS_DROP_SOURCE_LINKS
    app.config['SKIP_VOICE_NORMALIZE'] = Config.SKIP_VOICE_NORMALIZE
    app.config['TTS_DEBUG_LOG'] = Config.TTS_DEBUG_LOG
    # Request hardening + segment gap (read by routes at request time)
    app.config['MAX_INPUT_CHARS'] = Config.MAX_INPUT_CHARS
    app.config['API_KEY'] = Config.API_KEY
    app.config['SEGMENT_GAP_MS'] = Config.SEGMENT_GAP_MS
    app.config['DEFAULT_VOICE'] = Config.DEFAULT_VOICE
    app.config['FALLBACK_UNKNOWN_VOICE'] = Config.FALLBACK_UNKNOWN_VOICE

    # Apply overrides
    if config_overrides:
        app.config.update(config_overrides)

    # Register blueprints
    from app.api.routes import api

    app.register_blueprint(api)

    logger.info('Flask application created')

    return app


def init_tts_service(
    model_path: str = None,
    voices_dir: str = None,
    skip_voice_normalize: bool = False,
) -> None:
    """
    Initialize the TTS service with model and voices.

    Args:
        model_path: Optional path to model file
        voices_dir: Optional path to voices directory
        skip_voice_normalize: Skip startup WAV loudness normalization
    """
    from app.services.tts import get_tts_service

    logger = get_logger()
    tts = get_tts_service()

    # Load model
    tts.load_model(model_path)

    # Set voices directory
    if voices_dir:
        normalize_voice_wavs(voices_dir, skip=skip_voice_normalize)
        tts.set_voices_dir(voices_dir)
    else:
        # Fall back to the default data/voices directory when it exists
        default_voices = Config.DATA_ROOT / 'voices'
        if default_voices.is_dir():
            normalize_voice_wavs(str(default_voices), skip=skip_voice_normalize)
            tts.set_voices_dir(str(default_voices))

    logger.info('TTS service initialized')


def create_initialized_app(
    config_overrides: dict | None = None,
    model_path: str | None = None,
    voices_dir: str | None = None,
    skip_voice_normalize: bool = False,
) -> 'Flask':
    """Create the Flask app and initialize the shared TTS service."""
    app = create_app(config_overrides=config_overrides)
    init_tts_service(
        model_path=model_path,
        voices_dir=voices_dir,
        skip_voice_normalize=skip_voice_normalize,
    )
    return app
