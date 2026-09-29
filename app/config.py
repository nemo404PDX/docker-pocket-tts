"""
Configuration management for PocketTTS OpenAI Server.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

# Canonical default for the code-block placeholder lives in normalize.py (where
# TtsPipelineConfig consumes it). Importing here keeps a single source of truth
# instead of duplicating the literal across config + routes.
from app.services.normalize import DEFAULT_CODE_BLOCK_PHRASE, DEFAULT_HERMES_MARKER_PATTERN


def get_base_path() -> Path:
    """Get the base path for the application (repo root)."""
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


def _builtin_voice_names() -> list[str]:
    """Return pocket-tts' built-in voice names, derived from the installed package.

    Reads the private ``_ORIGINS_OF_PREDEFINED_VOICES`` mapping so this stays in
    sync with upstream automatically. Falls back to a minimal known set if that
    symbol is ever renamed or removed (keeps the server importable).
    """
    try:
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

        return sorted(_ORIGINS_OF_PREDEFINED_VOICES)
    except Exception:  # pragma: no cover - defensive fallback
        return [
            'alba',
            'marius',
            'javert',
            'jean',
            'fantine',
            'cosette',
            'eponine',
            'azelma',
        ]


class Config:
    """Application configuration loaded from environment variables."""

    # Base paths
    BASE_PATH = get_base_path()
    DATA_ROOT = get_data_root(BASE_PATH)

    # Server settings
    HOST = os.environ.get('POCKET_TTS_HOST', '0.0.0.0')
    PORT = int(os.environ.get('POCKET_TTS_PORT', '49112'))
    # Waitress worker threads (only used by the `python server.py` path; the
    # container path is driven by whatever WSGI runner the image invokes).
    THREADS = int(os.environ.get('POCKET_TTS_THREADS', '4'))

    # Request hardening
    # Reject synthesis requests whose input exceeds this many characters so a
    # single client can't pin the CPU with an unbounded payload. 0 disables the cap.
    MAX_INPUT_CHARS = int(os.environ.get('POCKET_TTS_MAX_INPUT_CHARS', str(32 * 1000)))
    # Optional bearer-token auth. When set, every API route (except /health and
    # /ready) requires `Authorization: Bearer <key>`. Unset/empty disables it so
    # existing local setups keep working unchanged.
    API_KEY = os.environ.get('POCKET_TTS_API_KEY') or None

    # Voice cache bound — cap how many loaded voice states we hold in memory so a
    # client cycling through many custom voices can't grow the cache without limit.
    MAX_CACHED_VOICES = int(os.environ.get('POCKET_TTS_MAX_CACHED_VOICES', '8'))

    # Model settings
    MODEL_PATH = os.environ.get('POCKET_TTS_MODEL_PATH', None)
    # HF cache dir. pocket-tts resolves weights from here first (offline-friendly):
    # pre-populate it with `hf download kyutai/pocket-tts` and the server loads
    # locally without hitting HuggingFace at startup.
    MODEL_CACHE_DIR = os.environ.get(
        'POCKET_TTS_MODEL_CACHE_DIR',
        str(DATA_ROOT / 'models' / 'pocket-tts'),
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

    # Silence inserted between synthesized segments (ms). Long text is split into
    # segments and concatenated; without a gap the sentences run together with no
    # breathing room. 0 disables the gap entirely. This is the FULL gap, applied only at
    # sentence-final boundaries (a segment ending in . ! ?) — see SEGMENT_SEAM_MS for the
    # smaller seam used between mid-sentence fragments so long clauses don't get a full
    # "breath" landing inside a phrase.
    SEGMENT_GAP_MS = int(os.environ.get('POCKET_TTS_SEGMENT_GAP_MS', '200'))

    # Silence inserted between MID-SENTENCE fragments (ms) — i.e. where a single sentence
    # was split by the char/token caps rather than at a real sentence boundary. Kept small
    # so long clauses flow as one continuous phrase instead of getting an audible pause in
    # the middle ("...your [pause] code"). 0 concatenates fragments with no seam at all.
    SEGMENT_SEAM_MS = int(os.environ.get('POCKET_TTS_SEGMENT_SEAM_MS', '30'))

    # Extra decoder frames to generate after pocket-tts detects end-of-speech (1 frame =
    # 80 ms at 12.5 Hz). Empty/unset uses pocket-tts's built-in guess (~3 frames), which is
    # short enough that some voices clip the final consonant/syllable. Set e.g. "6" for a
    # longer tail; 0 disables the extra tail entirely. Note: generation is stochastic at
    # temp=0.3, so this is a per-voice tuning lever rather than a guaranteed fix — see README.
    _frames_after_eos_raw = os.environ.get('POCKET_TTS_FRAMES_AFTER_EOS', '').strip()
    FRAMES_AFTER_EOS: int | None = (
        int(_frames_after_eos_raw) if _frames_after_eos_raw.lstrip('-').isdigit() else None
    )

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
        DEFAULT_CODE_BLOCK_PHRASE,
    )
    # Regex of client markers to strip from TTS text (whole line removed). Defaults to the
    # Hermes-specific pattern; override per-deployment if you feed a different client's output.
    TTS_HERMES_MARKER_PATTERN = os.environ.get(
        'POCKET_TTS_HERMES_MARKER_PATTERN',
        DEFAULT_HERMES_MARKER_PATTERN,
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

    # Built-in voice names, resolved by pocket-tts internally. Derived from the
    # installed package so we track upstream instead of drifting (the old hardcoded
    # list had 8; pocket-tts ships ~27). Falls back to a minimal set if the private
    # symbol is ever renamed/removed.
    BUILTIN_VOICES = _builtin_voice_names()

    # Supported audio extensions for custom voices
    VOICE_EXTENSIONS = ('.wav', '.mp3', '.flac')
