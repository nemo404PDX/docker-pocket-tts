"""
Configuration management for PocketTTS OpenAI Server.
Loads settings from environment variables with sensible defaults.
"""

import os
import sys
from pathlib import Path


def get_base_path() -> Path:
    """Get the base path for the application, handling PyInstaller frozen state."""
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            # One-file mode
            return Path(sys._MEIPASS)
        else:
            # One-dir mode
            return Path(sys.executable).parent
    return Path(__file__).parent.parent


def _running_in_container() -> bool:
    return Path('/.dockerenv').exists()


def get_data_root(base_path: Path) -> Path:
    """Prefer a mounted container data directory, then fall back to repo-local data."""
    configured_data_dir = os.environ.get('POCKET_TTS_DATA_DIR')
    if configured_data_dir:
        return Path(configured_data_dir).expanduser()

    container_data_dir = Path('/data')
    if _running_in_container() and container_data_dir.exists():
        return container_data_dir

    return base_path / 'data'


class Config:
    """Application configuration loaded from environment variables."""

    # Base paths
    BASE_PATH = get_base_path()
    DATA_ROOT = get_data_root(BASE_PATH)
    IS_FROZEN = getattr(sys, 'frozen', False)

    # Server settings
    HOST = os.environ.get('POCKET_TTS_HOST', '0.0.0.0')
    PORT = int(os.environ.get('POCKET_TTS_PORT', '49112'))

    # Model settings
    MODEL_PATH = os.environ.get('POCKET_TTS_MODEL_PATH', None)
    MODEL_CACHE_DIR = os.environ.get(
        'POCKET_TTS_MODEL_CACHE_DIR',
        str(DATA_ROOT / 'models' / 'pocket-tts'),
    )
    LOCAL_MODEL_DIR = os.environ.get(
        'POCKET_TTS_LOCAL_MODEL_DIR',
        str(DATA_ROOT / 'models' / 'kyutai' / 'pocket-tts'),
    )
    PREFER_LOCAL_CLONING_MODEL = (
        os.environ.get('POCKET_TTS_PREFER_LOCAL_CLONING_MODEL', 'true').lower() == 'true'
    )
    REQUIRE_VOICE_CLONING = (
        os.environ.get('POCKET_TTS_REQUIRE_VOICE_CLONING', 'false').lower() == 'true'
    )
    _DEFAULT_VOICES_DIR = DATA_ROOT / 'voices'
    VOICES_DIR = os.environ.get(
        'POCKET_TTS_VOICES_DIR',
        str(_DEFAULT_VOICES_DIR) if _DEFAULT_VOICES_DIR.is_dir() else None,
    )
    _LOCAL_DEFAULT_VOICE = 'default.wav' if (_DEFAULT_VOICES_DIR / 'default.wav').is_file() else 'alba'
    DEFAULT_VOICE = os.environ.get('POCKET_TTS_DEFAULT_VOICE', _LOCAL_DEFAULT_VOICE)
    FALLBACK_UNKNOWN_VOICE = (
        os.environ.get('POCKET_TTS_FALLBACK_UNKNOWN_VOICE', 'true').lower() == 'true'
    )

    # Streaming default
    STREAM_DEFAULT = os.environ.get('POCKET_TTS_STREAM_DEFAULT', 'false').lower() == 'true'

    # Text preprocessing default
    TEXT_PREPROCESS_DEFAULT = (
        os.environ.get('POCKET_TTS_TEXT_PREPROCESS_DEFAULT', 'false').lower() == 'true'
    )
    TTS_NORMALIZE_DEFAULT = (
        os.environ.get(
            'POCKET_TTS_NORMALIZE_DEFAULT',
            str(TEXT_PREPROCESS_DEFAULT).lower(),
        ).lower()
        == 'true'
    )
    TTS_SEGMENTATION_DEFAULT = os.environ.get('POCKET_TTS_SEGMENTATION_DEFAULT', 'none')
    TTS_SKIP_CODE_BLOCKS = os.environ.get('POCKET_TTS_SKIP_CODE_BLOCKS', 'true').lower() == 'true'
    TTS_CODE_BLOCK_PHRASE = os.environ.get(
        'POCKET_TTS_CODE_BLOCK_PHRASE',
        ', see the code in the conversation history,',
    )
    TTS_EMOJI_HANDLING = os.environ.get('POCKET_TTS_EMOJI_HANDLING', 'strip')
    TTS_BRIEFING_MODE = os.environ.get('POCKET_TTS_BRIEFING_MODE', 'false').lower() == 'true'
    TTS_DROP_SOURCE_LINKS = (
        os.environ.get('POCKET_TTS_DROP_SOURCE_LINKS', str(TTS_BRIEFING_MODE).lower()).lower()
        == 'true'
    )
    SKIP_VOICE_NORMALIZE = (
        os.environ.get('POCKET_TTS_SKIP_VOICE_NORMALIZE', 'false').lower() == 'true'
    )
    TTS_DEBUG_LOG = os.environ.get('POCKET_TTS_DEBUG_LOG', 'false').lower() == 'true'

    # Logging
    LOG_LEVEL = os.environ.get('POCKET_TTS_LOG_LEVEL', 'INFO')
    LOG_TO_FILE = os.environ.get('POCKET_TTS_LOG_TO_FILE', 'false').lower() == 'true'
    LOG_DIR = os.environ.get('POCKET_TTS_LOG_DIR', str(DATA_ROOT / 'logs'))
    LOG_FILE = os.environ.get('POCKET_TTS_LOG_FILE', 'pocket_tts.log')
    LOG_MAX_BYTES = int(os.environ.get('POCKET_TTS_LOG_MAX_BYTES', str(10 * 1024 * 1024)))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('POCKET_TTS_LOG_BACKUP_COUNT', '5'))

    # Built-in voice mappings (these are resolved by pocket-tts internally)
    BUILTIN_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

    # Supported audio extensions for custom voices
    VOICE_EXTENSIONS = ('.wav', '.mp3', '.flac', '.safetensors')

    @classmethod
    def get_bundle_paths(cls) -> tuple:
        """Get bundled paths for frozen executables."""
        if cls.IS_FROZEN:
            voices_dir = cls.BASE_PATH / 'voices'
            model_path = cls.BASE_PATH / 'model' / 'b6369a24.yaml'
            return (
                str(voices_dir) if voices_dir.is_dir() else None,
                str(model_path) if model_path.is_file() else None,
            )
        return None, None

    @classmethod
    def get_template_folder(cls) -> str:
        """Get the templates folder path."""
        return str(cls.BASE_PATH / 'templates')

    @classmethod
    def get_static_folder(cls) -> str:
        """Get the static files folder path."""
        return str(cls.BASE_PATH / 'static')
