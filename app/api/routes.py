"""
Flask routes for the OpenAI-compatible TTS API.
"""

import hmac
import time

import torch
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    send_file,
    stream_with_context,
)

from app.logging_config import get_logger
from app.services.audio import (
    convert_audio,
    get_mime_type,
    is_supported_format,
    supported_formats,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.normalize import TtsPipelineConfig, segment_tts_text
from app.services.tts import get_tts_service

logger = get_logger('routes')

# Create blueprint
api = Blueprint('api', __name__)

# OpenAI-style voice ids often used by clients like Open WebUI.
# Map them to this server's default voice to avoid hard failures.
OPENAI_VOICE_ALIASES = {'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'zoe'}
VALID_SEGMENTATION_MODES = {'none', 'paragraph', 'sentence'}
VALID_EMOJI_HANDLING = {'strip', 'describe', 'keep'}


def _first_builtin_voice(tts) -> str:
    """Return a stable builtin fallback voice."""
    for voice in tts.list_voices():
        if voice.get('type') == 'builtin':
            return voice['id']
    return 'alba'


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return default


def _validated_choice(value, valid: set[str], default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized if normalized in valid else default


# Routes that must stay reachable without auth (container health/readiness probes).
_AUTH_EXEMPT_ROUTES = {'/health', '/ready'}


def _check_api_key():
    """Optional bearer-token guard.

    When ``API_KEY`` is set, every route except the health/readiness probes must
    present a matching ``Authorization: Bearer <key>`` header. Comparison uses a
    constant-time check to avoid leaking the key via timing. Returns an error
    response tuple when auth fails, or None when the request may proceed.
    """
    expected = current_app.config.get('API_KEY')
    if not expected:
        return None

    path = request.path.rstrip('/') or '/'
    if path in _AUTH_EXEMPT_ROUTES:
        return None

    provided = (request.headers.get('Authorization') or '').strip()
    # Accept "Bearer <key>" (case-insensitive scheme) or a bare key.
    token = provided[7:].strip() if provided[:7].lower() == 'bearer ' else provided
    if not hmac.compare_digest(token, expected):
        return jsonify(
            {
                'error': {
                    'message': 'Invalid or missing API key',
                    'type': 'authentication_error',
                    'code': None,
                }
            }
        ), 401


def _openai_error(message: str, err_type: str = 'invalid_request_error', code=None):
    """Build an OpenAI-shaped error envelope so standard clients parse it cleanly."""
    return jsonify({'error': {'message': message, 'type': err_type, 'code': code}})


# Enforce the optional API key on every route (health/readiness are exempt).
api.before_request(_check_api_key)


def _build_pipeline_config(data: dict, app_config) -> TtsPipelineConfig:
    briefing_mode = app_config.get('TTS_BRIEFING_MODE', False)
    normalize_default = app_config.get('TTS_NORMALIZE_DEFAULT', False)
    segmentation_default = app_config.get('TTS_SEGMENTATION_DEFAULT', 'none')
    skip_code_blocks_default = app_config.get('TTS_SKIP_CODE_BLOCKS', True)
    # Single source of truth for the code-block placeholder is Config (which reads
    # normalize.DEFAULT_CODE_BLOCK_PHRASE); fall back to it if not in app config.
    from app.config import Config

    code_block_phrase_default = app_config.get(
        'TTS_CODE_BLOCK_PHRASE',
        Config.TTS_CODE_BLOCK_PHRASE,
    )
    # Client-marker regex (server-level default; overridable per-request). Single source of
    # truth is Config (which reads normalize.DEFAULT_HERMES_MARKER_PATTERN); fall back if absent.
    marker_pattern_default = app_config.get(
        'TTS_HERMES_MARKER_PATTERN',
        Config.TTS_HERMES_MARKER_PATTERN,
    )
    emoji_default = app_config.get('TTS_EMOJI_HANDLING', 'strip')
    drop_source_links_default = app_config.get('TTS_DROP_SOURCE_LINKS', briefing_mode)

    effective_normalize_default = True if briefing_mode else normalize_default
    effective_segmentation_default = 'sentence' if briefing_mode else segmentation_default

    segmentation = _validated_choice(
        data.get('segmentation'),
        VALID_SEGMENTATION_MODES,
        effective_segmentation_default,
    )
    normalize = _coerce_bool(data.get('normalize'), effective_normalize_default)

    return TtsPipelineConfig(
        normalize=normalize,
        skip_code_blocks=_coerce_bool(data.get('skip_code_blocks'), skip_code_blocks_default),
        code_block_phrase=str(data.get('code_block_phrase') or code_block_phrase_default),
        emoji_handling=_validated_choice(
            data.get('emoji_handling'),
            VALID_EMOJI_HANDLING,
            emoji_default,
        ),
        segmentation=segmentation,
        drop_source_links_section=_coerce_bool(
            data.get('drop_source_links_section'),
            drop_source_links_default,
        ),
        # Client-marker regex. Request may override the server/env default; an explicit empty
        # value disables marker stripping, and absence falls back to the configured default.
        marker_pattern=(
            marker_pattern_default if 'marker_pattern' not in data else str(data.get('marker_pattern'))
        ),
    )


def _log_pipeline_debug(data: dict, pipeline_config: TtsPipelineConfig, segments: list[str]) -> None:
    logger.debug(
        'TTS pipeline config normalize=%s segmentation=%s skip_code_blocks=%s emoji_handling=%s '
        'drop_source_links=%s '
        'segment_count=%s input_len=%s segment_lens=%s preview=%s',
        pipeline_config.normalize,
        pipeline_config.segmentation,
        pipeline_config.skip_code_blocks,
        pipeline_config.emoji_handling,
        pipeline_config.drop_source_links_section,
        len(segments),
        len(str(data.get('input') or data.get('text') or '')),
        [len(segment) for segment in segments[:8]],
        segments[:2],
    )


@api.route('/')
def home():
    """Service info endpoint."""
    return (
        jsonify(
            {
                'status': 'ok',
                'service': 'pocket-tts',
                'supported_formats': sorted(supported_formats()),
            }
        ),
        200,
    )


@api.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.

    Returns service status and basic model info.
    """
    tts = get_tts_service()

    # Validate a built-in voice quickly
    voice_valid, voice_msg = tts.validate_voice('alba')

    return jsonify(
        {
            'status': 'healthy' if tts.is_loaded else 'unhealthy',
            'model_loaded': tts.is_loaded,
            'device': tts.device if tts.is_loaded else None,
            'sample_rate': tts.sample_rate if tts.is_loaded else None,
            'voices_dir': tts.voices_dir,
            'voice_check': {'valid': voice_valid, 'message': voice_msg},
        }
    ), 200 if tts.is_loaded else 503


@api.route('/ready', methods=['GET'])
def ready():
    """Readiness check for environments that distinguish liveness from readiness."""
    tts = get_tts_service()

    if not tts.is_loaded:
        return jsonify({'ready': False, 'model_loaded': False}), 503

    return jsonify(
        {
            'ready': True,
            'model_loaded': True,
            'device': tts.device,
            'sample_rate': tts.sample_rate,
        }
    ), 200


@api.route('/debug/normalize', methods=['POST'])
def debug_normalize_endpoint():
    """Debug endpoint: show exactly what preprocessing does to input text.

    Disabled unless POCKET_TTS_DEBUG_LOG=true (it echoes request content back).
    """
    if not current_app.config.get('TTS_DEBUG_LOG'):
        return jsonify({'error': 'debug endpoints disabled; set POCKET_TTS_DEBUG_LOG=true'}), 404
    from app.services.normalize import normalize_tts_text, segment_tts_text, TtsPipelineConfig
    data = request.get_json(silent=True) or {}
    text = data.get('input') or data.get('text', '')
    config = _build_pipeline_config(data, current_app.config)
    normalized = normalize_tts_text(text, config)
    segments = segment_tts_text(text, config)
    return jsonify({
        'original_length': len(text),
        'normalized_length': len(normalized),
        'normalized_preview': normalized[:500],
        'segment_count': len(segments),
        'segments': segments[:10],
    }), 200


@api.route('/v1/voices', methods=['GET'])
def list_voices():
    """
    List available voices.

    Returns OpenAI-compatible voice list format.
    """
    tts = get_tts_service()
    voices = tts.list_voices()

    return jsonify(
        {
            'object': 'list',
            'data': [
                {
                    'id': v['id'],
                    'name': v['name'],
                    'object': 'voice',
                    'type': v.get('type', 'builtin'),
                    # OpenAI-compatible field. Built-ins are 0 (part of the model); custom
                    # voices carry their file mtime so clients can sort/dedupe stably.
                    'created_at': v.get('created_at', 0),
                }
                for v in voices
            ],
        }
    )


@api.route('/tts', methods=['POST'])
@api.route('/audio', methods=['POST'])
@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (ignored, for compatibility)
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        stream: boolean (optional) - Enable streaming

    Returns:
        Audio file or streaming audio response
    """
    data = request.get_json(silent=True)

    if not data:
        return _openai_error('Missing JSON body', code='invalid_request'), 400

    text = data.get('input') or data.get('text')
    if not text:
        return _openai_error("Missing 'input' text", code='missing_input'), 400

    # Reject unreasonably large payloads before they reach the model. A single
    # client could otherwise pin the CPU by POSTing megabytes of text.
    max_chars = current_app.config.get('MAX_INPUT_CHARS', 0) or 0
    if max_chars and len(text) > max_chars:
        logger.warning(
            'Rejecting TTS request: input %d chars exceeds cap %d', len(text), max_chars
        )
        return (
            _openai_error(
                f"Input too long: {len(text)} characters exceeds the maximum of "
                f"{max_chars}. Split your text into smaller requests.",
                code='input_too_long',
            ),
            413,
        )

    voice = data.get('voice') or current_app.config.get('DEFAULT_VOICE', 'alba')
    if isinstance(voice, str) and voice.lower() in OPENAI_VOICE_ALIASES:
        voice = current_app.config.get('DEFAULT_VOICE', 'alba')
    stream_request = data.get('stream', False)

    response_format = data.get('response_format', 'mp3')
    target_format = validate_format(response_format)

    # Reject formats this environment can't encode (e.g. aac/opus without an ffmpeg
    # backend, or unknown strings) with 400 instead of failing mid-generation with 500.
    if not is_supported_format(target_format):
        return (
            _openai_error(
                f"Unsupported response_format '{response_format}'. "
                f"Supported formats: {', '.join(sorted(supported_formats()))}.",
                code='unsupported_format',
            ),
            400,
        )

    # Request metadata (no text content — user input is arbitrary)
    logger.info(
        'TTS request: len=%d stream=%s voice=%s format=%s',
        len(text),
        stream_request,
        voice,
        target_format,
    )

    tts = get_tts_service()

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    default_voice = current_app.config.get('DEFAULT_VOICE', 'alba')
    fallback_unknown_voice = current_app.config.get('FALLBACK_UNKNOWN_VOICE', True)

    if not is_valid and fallback_unknown_voice and voice != default_voice:
        logger.warning("Unknown voice '%s'; falling back to default voice '%s'.", voice, default_voice)
        voice = default_voice
        is_valid, msg = tts.validate_voice(voice)

    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        logger.warning("Invalid voice '%s' requested. Available sample: %s", voice, available[:5])
        return (
            jsonify(
                {
                    'error': {
                        'message': f"Voice '{voice}' not found",
                        'type': 'invalid_request_error',
                        'code': 'unknown_voice',
                    },
                    # Helpful extras beyond the OpenAI envelope:
                    'available_voices': available[:10],  # Limit to first 10
                    'hint': 'Use /v1/voices to see all available voices',
                }
            ),
            400,
        )

    try:
        try:
            voice_state = tts.get_voice_state(voice)
        except ValueError as e:
            # If cloning weights are unavailable, fall back to a builtin voice
            # so clients can still synthesize audio.
            err_msg = str(e).lower()
            if 'voice cloning' in err_msg and fallback_unknown_voice:
                fallback_voice = _first_builtin_voice(tts)
                logger.warning(
                    "Voice '%s' requires cloning model; falling back to builtin '%s'.",
                    voice,
                    fallback_voice,
                )
                voice = fallback_voice
                voice_state = tts.get_voice_state(voice)
            else:
                raise

        # Check if streaming should be used
        use_streaming = stream_request or current_app.config.get('STREAM_DEFAULT', False)

        # Streaming supports only PCM/WAV today; fall back to file for other formats.
        if use_streaming and target_format not in ('pcm', 'wav'):
            logger.warning(
                "Streaming format '%s' is not supported; returning full file instead.",
                target_format,
            )
            use_streaming = False
        pipeline_config = _build_pipeline_config(data, current_app.config)
        segments = segment_tts_text(text, pipeline_config)
        # Log preprocessing result for debugging
        logger.info(
            'TTS segments: count=%d lens=%s',
            len(segments),
            [len(s) for s in segments[:8]],
        )
        if not segments:
            # Everything was stripped by preprocessing — fall back to a brief phrase
            # instead of hard-failing with 422 (e.g., messages that are just terminal markers)
            # Skip fallback for streaming deltas — partial text is expected to be empty after stripping
            if stream_request:
                logger.info(
                    'Empty streaming delta (%d chars input); skipping synthesis',
                    len(text),
                )
                return Response(b'', mimetype=f'audio/{target_format}', status=200)
            logger.warning(
                'No TTS content after preprocessing (%d chars input); falling back to placeholder',
                len(text),
            )
            segments = ['No content']

        if use_streaming:
            return _stream_audio(tts, voice_state, segments, target_format)
        return _generate_file(tts, voice_state, segments, target_format)

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return _openai_error(str(e), err_type='invalid_request_error', code='voice_load_failed'), 400
    except Exception as e:
        logger.exception('Generation failed')
        return _openai_error(
            f'Speech generation failed: {e}', err_type='server_error', code='generation_failed'
        ), 500


def _concatenate_audio_tensors(audio_tensors: list[torch.Tensor]) -> torch.Tensor:
    normalized: list[torch.Tensor] = []
    for tensor in audio_tensors:
        if tensor.dim() == 1:
            normalized.append(tensor.unsqueeze(0))
            continue
        if tensor.dim() != 2:
            raise ValueError(f'Expected 1D or 2D audio tensor, got shape {tuple(tensor.shape)}')
        normalized.append(tensor)

    combined = torch.cat(normalized, dim=-1)
    return combined.squeeze(0) if combined.shape[0] == 1 else combined


def _silence_tensor(tts, gap_ms: int, channels: int = 1) -> torch.Tensor | None:
    """Return a zero-filled silence tensor of ``gap_ms`` duration, or None if disabled.

    This is the "breathing room" inserted between synthesized segments so long text
    doesn't run together with no pause. Built as plain zeros (no model call), matching
    the model's sample rate and channel count.
    """
    if gap_ms <= 0:
        return None
    n = max(1, int(round(tts.sample_rate * gap_ms / 1000.0)))
    shape = (channels, n) if channels > 1 else (n,)
    return torch.zeros(shape, dtype=torch.float32)


def _boundary_gap_ms(segment: str) -> int:
    """Silence to insert AFTER ``segment`` before the next one is synthesized.

    A segment that ends a sentence (., !, ? — ignoring trailing quotes/brackets) gets
    the full SEGMENT_GAP_MS "breath". A mid-sentence fragment (split by the char/token
    caps rather than at a real boundary) gets the smaller SEGMENT_SEAM_MS so long clauses
    flow as one continuous phrase instead of getting an audible pause in the middle
    ("...your [pause] code"). The seam is capped at the full gap so a sentence boundary
    never has less silence than a mid-clause split, and both collapse to 0 when the full
    gap is disabled (SEGMENT_GAP_MS=0).
    """
    gap_ms = current_app.config.get('SEGMENT_GAP_MS', 0) or 0
    seam_ms = current_app.config.get('SEGMENT_SEAM_MS', 0) or 0
    # Ignore trailing quotes/brackets when detecting the boundary (He said "stop." → stop).
    stripped = segment.rstrip().rstrip("\"')]").strip()
    if stripped and stripped[-1] in '.!?':
        return gap_ms
    return min(seam_ms, gap_ms)


def _synthesize_segments(tts, voice_state, segments: list[str]) -> torch.Tensor:
    audio_tensors = [tts.generate_audio(voice_state, segment) for segment in segments]
    if len(audio_tensors) == 1:
        return audio_tensors[0]

    # Insert boundary-aware silence between segments (see _boundary_gap_ms): a full breath
    # at sentence-final boundaries, a small seam between mid-sentence fragments.
    first_channels = audio_tensors[0].shape[0] if audio_tensors[0].dim() == 2 else 1
    combined: list[torch.Tensor] = []
    for i, tensor in enumerate(audio_tensors):
        if i > 0:
            silence = _silence_tensor(tts, _boundary_gap_ms(segments[i - 1]), channels=first_channels)
            if silence is not None:
                combined.append(silence)
        combined.append(tensor)
    return _concatenate_audio_tensors(combined)


def _generate_file(tts, voice_state, segments: list[str], fmt: str):
    """Generate complete audio and return as file."""
    t0 = time.time()
    audio_tensor = _synthesize_segments(tts, voice_state, segments)
    generation_time = time.time() - t0
    char_count = sum(len(segment) for segment in segments)
    duration_sec = audio_tensor.shape[-1] / tts.sample_rate

    logger.info(
        'Generated %s chars across %s segment(s) into %.2fs audio in %.2fs',
        char_count,
        len(segments),
        duration_sec,
        generation_time,
    )

    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    mimetype = get_mime_type(fmt)

    return send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )


def _stream_audio(tts, voice_state, segments: list[str], fmt: str):
    """Stream audio chunks."""
    # Normalize streaming format: we always emit PCM bytes, optionally wrapped
    # in a WAV container. For non-PCM/WAV formats (e.g. mp3, opus), coerce to
    # raw PCM to avoid mismatched content-type vs. payload.
    stream_fmt = fmt
    if stream_fmt not in ('pcm', 'wav'):
        logger.warning(
            "Requested streaming format '%s' is not supported for streaming; "
            "falling back to 'pcm'.",
            stream_fmt,
        )
        stream_fmt = 'pcm'

    def generate():
        total_chars = sum(len(segment) for segment in segments)
        logger.info(
            'Starting streaming generation for %s chars across %d segment(s)',
            total_chars,
            len(segments),
        )
        # Insert boundary-aware silence between segments (see _boundary_gap_ms): a full
        # breath at sentence-final boundaries, a small seam between mid-sentence fragments.
        silence_cache: dict[int, bytes] = {}

        def _silence_for(ms: int) -> bytes | None:
            if ms <= 0:
                return None
            if ms not in silence_cache:
                silence_cache[ms] = tensor_to_pcm_bytes(_silence_tensor(tts, ms))
            return silence_cache[ms]

        for i, segment in enumerate(segments):
            if i > 0:
                silence_pcm = _silence_for(_boundary_gap_ms(segments[i - 1]))
                if silence_pcm is not None:
                    yield silence_pcm
            stream = tts.generate_audio_stream(voice_state, segment)
            for chunk_tensor in stream:
                yield tensor_to_pcm_bytes(chunk_tensor)

    def stream_with_header():
        # Yield WAV header first if streaming as WAV
        if stream_fmt == 'wav':
            yield write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
        yield from generate()

    mimetype = get_mime_type(stream_fmt)

    return Response(stream_with_context(stream_with_header()), mimetype=mimetype)
