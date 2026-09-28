"""
TTS Service - handles model loading, voice management, and audio generation.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from app.config import Config
from app.logging_config import get_logger

logger = get_logger('tts')

# Lazy import pocket_tts to allow for better error handling
TTSModel = None


def _safe_path(path_value: str) -> Path:
    """Return an absolute path without following symlinks (avoids symlink-loop crashes)."""
    path = Path(path_value).expanduser()
    return Path(os.path.abspath(path))


def _ensure_pocket_tts():
    """Ensure pocket-tts is imported."""
    global TTSModel
    if TTSModel is None:
        try:
            from pocket_tts import TTSModel as _TTSModel

            TTSModel = _TTSModel
        except ImportError as exc:
            raise ImportError('pocket-tts not found. Install with: pip install pocket-tts') from exc


def _configure_hf_cache() -> None:
    """Route Hugging Face cache/downloads into the configured local model directory."""
    cache_dir = _safe_path(Config.MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Align common HF-related cache vars to one project-local location.
    os.environ['POCKET_TTS_MODEL_CACHE_DIR'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)

    # Ensure HF auth token is visible to huggingface_hub.
    # Prefer env vars, but optionally hydrate them from a Docker secret file.
    hf_token = _load_hf_token()
    if hf_token:
        os.environ.setdefault('HF_TOKEN', hf_token)
        if not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token


def _load_hf_token() -> str | None:
    """Load the HF token from env vars or an optional secrets file."""
    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if hf_token:
        return hf_token

    token_file = os.environ.get('HF_TOKEN_FILE', '/run/secrets/hf_token').strip()
    if not token_file:
        return None

    token_path = Path(token_file).expanduser()
    if not token_path.is_file():
        return None

    try:
        secret = token_path.read_text(encoding='utf-8').strip()
    except OSError as exc:
        logger.warning('Could not read HF token file %s: %s', token_path, exc)
        return None

    return secret or None


def _build_local_cloning_config() -> str | None:
    """
    Build a local PocketTTS config that points to locally downloaded cloning weights.

    Expected supported layouts:
      <MODEL_CACHE_DIR>/models--kyutai--pocket-tts/snapshots/<rev>/tts_b6369a24.safetensors
      <LOCAL_MODEL_DIR>/tts_b6369a24.safetensors
    """
    if not Config.PREFER_LOCAL_CLONING_MODEL:
        return None

    cache_root = _safe_path(Config.MODEL_CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)

    local_weights = None
    local_tokenizer = None

    # Layout 1: direct local-dir download (e.g. hf download --local-dir)
    local_repo_dir = _safe_path(Config.LOCAL_MODEL_DIR)
    direct_weights = local_repo_dir / 'tts_b6369a24.safetensors'
    direct_tokenizer = local_repo_dir / 'tokenizer.model'
    if direct_weights.is_file():
        local_weights = direct_weights
        if direct_tokenizer.is_file():
            local_tokenizer = direct_tokenizer

    # Layout 2: huggingface_hub snapshots cache
    if local_weights is None:
        snapshots_root = cache_root / 'models--kyutai--pocket-tts' / 'snapshots'
        if snapshots_root.is_dir():
            candidates = sorted(
                snapshots_root.glob('*/tts_b6369a24.safetensors'),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                local_weights = _safe_path(str(candidates[0]))
                tokenizer_candidate = local_weights.parent / 'tokenizer.model'
                if tokenizer_candidate.is_file():
                    local_tokenizer = _safe_path(str(tokenizer_candidate))

    try:
        import pocket_tts as pocket_tts_pkg
    except ImportError:
        return None

    default_config = Path(pocket_tts_pkg.__file__).parent / 'config' / 'b6369a24.yaml'
    if not default_config.is_file():
        return None

    local_config = cache_root / 'local-b6369a24.yaml'
    config_text = default_config.read_text(encoding='utf-8')
    if local_weights is not None:
        preferred_weights_path = local_weights.as_posix()
    else:
        # Prefer the cloning repo directly when local files are not present.
        preferred_weights_path = (
            'hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b'
        )

    if local_tokenizer is not None:
        preferred_tokenizer_path = local_tokenizer.as_posix()
    else:
        preferred_tokenizer_path = 'hf://kyutai/pocket-tts/tokenizer.model'

    config_text = config_text.replace(
        'weights_path: hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b',
        f'weights_path: {preferred_weights_path}',
    )
    config_text = config_text.replace(
        'tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3',
        f'tokenizer_path: {preferred_tokenizer_path}',
    )
    local_config.write_text(config_text, encoding='utf-8')
    return str(local_config)


class TTSService:
    """
    Service class for Text-to-Speech operations.
    Manages model loading, voice caching, and audio generation.
    """

    def __init__(self):
        self.model = None
        self.voice_cache: dict = {}
        self.voices_dir: str | None = None
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded and self.model is not None

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self.model:
            return self.model.sample_rate
        return 24000  # Default pocket-tts sample rate

    @property
    def device(self) -> str:
        """Get the model's device."""
        if self.model:
            return str(self.model.device)
        return 'unknown'

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load the TTS model.

        Args:
            model_path: Optional path to model file or variant name
        """
        _configure_hf_cache()
        _ensure_pocket_tts()

        logger.info('Loading Pocket TTS model...')
        logger.info('Using model cache directory: %s', os.environ.get('HUGGINGFACE_HUB_CACHE'))
        t0 = time.time()

        # Determine model path
        effective_path = model_path

        if not effective_path:
            preferred_config = _build_local_cloning_config()
            if preferred_config:
                effective_path = preferred_config
                logger.info('Using preferred cloning model config: %s', effective_path)

        if not effective_path:
            logger.info('Loading default model from HuggingFace...')

        try:
            if effective_path:
                logger.info(f'Loading model from: {effective_path}')
                self.model = TTSModel.load_model(config=effective_path)
            else:
                logger.info('Loading default model from HuggingFace...')
                self.model = TTSModel.load_model()

            if Config.REQUIRE_VOICE_CLONING and not getattr(self.model, 'has_voice_cloning', True):
                raise RuntimeError(
                    'Voice-cloning weights were not loaded. '
                    'Set a valid Hugging Face token/access and retry.'
                )

            self._model_loaded = True
            load_time = time.time() - t0
            logger.info(
                f'Model loaded in {load_time:.2f}s. '
                f'Device: {self.device}, Sample Rate: {self.sample_rate}'
            )

        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise

    def set_voices_dir(self, voices_dir: str | None) -> None:
        """
        Set the directory for custom voice files.

        Args:
            voices_dir: Path to directory containing voice files
        """
        if voices_dir and os.path.isdir(voices_dir):
            self.voices_dir = voices_dir
            logger.info(f'Voices directory set to: {voices_dir}')
        elif voices_dir:
            logger.warning(f'Voices directory not found: {voices_dir}')
            self.voices_dir = None
        else:
            self.voices_dir = None

    def get_voice_state(self, voice_id_or_path: str) -> dict:
        """
        Resolve voice ID to a model state with caching.

        Args:
            voice_id_or_path: Voice identifier (name, file path, or URL)

        Returns:
            Model state dictionary for the voice

        Raises:
            ValueError: If voice cannot be loaded
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        # Resolve the voice path
        resolved_key = self._resolve_voice_path(voice_id_or_path)

        # Check cache
        if resolved_key in self.voice_cache:
            logger.debug(f'Using cached voice state for: {resolved_key}')
            return self.voice_cache[resolved_key]

        # Load voice
        logger.info(f'Loading voice: {resolved_key}')
        t0 = time.time()

        try:
            state = self.model.get_state_for_audio_prompt(resolved_key)
            self.voice_cache[resolved_key] = state
            load_time = time.time() - t0
            logger.info(f'Voice loaded in {load_time:.2f}s: {resolved_key}')
            return state

        except Exception as e:
            logger.error(f"Failed to load voice '{voice_id_or_path}': {e}")
            raise ValueError(f"Voice '{voice_id_or_path}' could not be loaded: {e}") from e

    def _resolve_voice_path(self, voice_id_or_path: str) -> str:
        """
        Resolve a voice identifier to its actual path or ID.

        Args:
            voice_id_or_path: Voice identifier

        Returns:
            Resolved path or identifier

        Raises:
            ValueError: If unsafe URL scheme is used
        """
        # Block potentially dangerous URL schemes (SSRF protection)
        if voice_id_or_path.startswith(('http://', 'https://')):
            raise ValueError(
                f'URL scheme not allowed for security reasons: {voice_id_or_path[:50]}. '
                "Use 'hf://' for HuggingFace models or provide a local file path."
            )

        # Allow HuggingFace URLs
        if voice_id_or_path.startswith('hf://'):
            return voice_id_or_path

        # Check if it's a built-in voice
        if voice_id_or_path.lower() in Config.BUILTIN_VOICES:
            return voice_id_or_path.lower()

        # Check voices directory
        if self.voices_dir:
            for ext in Config.VOICE_EXTENSIONS:
                # Try exact match first
                possible_path = os.path.join(self.voices_dir, voice_id_or_path)
                if os.path.exists(possible_path):
                    return os.path.abspath(possible_path)

                # Try with extension
                if not voice_id_or_path.endswith(ext):
                    possible_path = os.path.join(self.voices_dir, voice_id_or_path + ext)
                    if os.path.exists(possible_path):
                        return os.path.abspath(possible_path)

        # Check if it's an absolute path that exists
        if os.path.isabs(voice_id_or_path) and os.path.exists(voice_id_or_path):
            return voice_id_or_path

        # Return as-is, let pocket-tts handle it
        return voice_id_or_path

    def validate_voice(self, voice_id_or_path: str) -> tuple[bool, str]:
        """
        Validate if a voice can be loaded (fast check without full loading).

        Args:
            voice_id_or_path: Voice identifier

        Returns:
            Tuple of (is_valid, message)
        """
        # Block unsafe URL schemes first
        if voice_id_or_path.startswith(('http://', 'https://')):
            return (
                False,
                'HTTP/HTTPS URLs are not allowed for security reasons. Use hf:// for HuggingFace models.',
            )

        try:
            resolved = self._resolve_voice_path(voice_id_or_path)
        except ValueError as e:
            return False, str(e)

        # Built-in voices are always valid
        if resolved.lower() in Config.BUILTIN_VOICES:
            return True, f'Built-in voice: {resolved}'

        # HuggingFace URLs - assume valid
        if resolved.startswith('hf://'):
            return True, f'HuggingFace voice: {resolved}'

        # Local file - check existence
        if os.path.exists(resolved):
            return True, f'Local voice file: {resolved}'

        return False, f'Voice not found: {voice_id_or_path}'

    def generate_audio(self, voice_state: dict, text: str) -> torch.Tensor:
        """
        Generate complete audio for given text.

        Args:
            voice_state: Model state from get_voice_state()
            text: Text to synthesize

        Returns:
            Audio tensor
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        t0 = time.time()
        audio = self.model.generate_audio(voice_state, text)
        gen_time = time.time() - t0

        logger.info(f'Generated {len(text)} chars in {gen_time:.2f}s')
        return audio

    def generate_audio_stream(self, voice_state: dict, text: str) -> Iterator[torch.Tensor]:
        """
        Generate audio in streaming chunks.

        Args:
            voice_state: Model state from get_voice_state()
            text: Text to synthesize

        Yields:
            Audio tensor chunks
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        logger.info(f'Starting streaming generation for {len(text)} chars')
        yield from self.model.generate_audio_stream(voice_state, text)

    def list_voices(self) -> list[dict]:
        """
        List all available voices.

        Returns:
            List of voice dictionaries with 'id' and 'name' keys
        """
        voices = []

        # Built-in voices (sorted alphabetically)
        builtin_sorted = sorted(Config.BUILTIN_VOICES)
        for voice in builtin_sorted:
            voices.append({'id': voice, 'name': voice.capitalize(), 'type': 'builtin'})

        # Custom voices from directory
        custom_voices = []
        if self.voices_dir and os.path.isdir(self.voices_dir):
            voice_dir = Path(self.voices_dir)

            # Collect all valid files
            voice_files = []
            for ext in Config.VOICE_EXTENSIONS:
                voice_files.extend(voice_dir.glob(f'*{ext}'))

            # Sort alphabetically by filename
            voice_files.sort(key=lambda f: f.name.lower())

            for voice_file in voice_files:
                # Format name: "bobby_mcfern" -> "Bobby Mcfern"
                clean_name = voice_file.stem.replace('_', ' ').replace('-', ' ').title()

                custom_voices.append(
                    {
                        'id': voice_file.name,
                        'name': clean_name,
                        'type': 'custom',
                    }
                )

        voices.extend(custom_voices)
        return voices


# Global service instance
_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    """Get the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
