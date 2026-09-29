"""
Audio conversion and streaming utilities.
"""

import io
import math
import struct
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from app.logging_config import get_logger

logger = get_logger('audio')

# Target loudness for startup voice normalization, and the tolerance band within which a
# file is left untouched. Used as defaults by normalize_voice_wavs().
TARGET_VOICE_DBFS = -20.0
VOICE_NORMALIZE_TOLERANCE_DB = 3.0

# Formats torchaudio can encode with the soundfile backend alone (always available).
_ALWAYS_ENCODABLE = {'wav', 'mp3', 'flac'}


def supported_formats() -> set[str]:
    """Return the audio formats this environment can actually encode.

    wav/mp3/flac always work (soundfile backend); aac/opus additionally require an
    ffmpeg backend; pcm is produced natively by us (raw 16-bit PCM, no container).
    Caching keeps this cheap — backends don't change at runtime.
    """
    cached = getattr(supported_formats, '_cache', None)
    if cached is not None:
        return cached

    formats = set(_ALWAYS_ENCODABLE) | {'pcm'}
    try:
        has_ffmpeg = 'ffmpeg' in torchaudio.list_audio_backends()
    except Exception:  # pragma: no cover - defensive
        has_ffmpeg = False
    if has_ffmpeg:
        formats |= {'aac', 'opus'}

    supported_formats._cache = formats
    return formats


def validate_format(fmt: str) -> str:
    """
    Normalize a requested audio format (lowercase + the mpeg→mp3 alias).

    Does NOT check encodability — callers should also test membership in
    supported_formats() and reject unsupported requests with HTTP 400.
    """
    fmt = fmt.lower()

    # OpenAI sometimes sends 'mpeg' for mp3
    if fmt == 'mpeg':
        return 'mp3'

    return fmt


def is_supported_format(fmt: str) -> bool:
    """True if ``fmt`` (already normalized via validate_format) can be encoded here."""
    return fmt in supported_formats()


def convert_audio(
    audio_tensor: torch.Tensor, sample_rate: int, target_format: str = 'wav'
) -> io.BytesIO:
    """
    Convert a raw audio tensor to a byte buffer in the specified format.

    Args:
        audio_tensor: The audio waveform (1D or 2D)
        sample_rate: The sample rate of the audio
        target_format: The target audio format

    Returns:
        Buffer containing the encoded audio data
    """
    buffer = io.BytesIO()

    # Ensure tensor is CPU
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    # Ensure 2D (channels, time)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.dim() != 2:
        raise ValueError(f'Expected audio tensor with 1 or 2 dims, got shape {tuple(audio_tensor.shape)}')

    # Raw PCM has no container — produce it natively (torchaudio can't encode 'pcm').
    if target_format == 'pcm':
        return io.BytesIO(tensor_to_pcm_bytes(audio_tensor))

    try:
        torchaudio.save(buffer, audio_tensor, sample_rate, format=target_format)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(
            'Error converting audio to %s (shape=%s, dtype=%s, sample_rate=%s): %s. '
            "Ensure the required codec/backend is installed for this format.",
            target_format,
            tuple(audio_tensor.shape),
            audio_tensor.dtype,
            sample_rate,
            e,
        )
        raise


def write_wav_header(
    sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, num_frames: int = 0
) -> bytes:
    """
    Generate a WAV header for streaming.

    If num_frames is 0, set to max value (streaming/unknown length).

    Args:
        sample_rate: Audio sample rate
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample
        num_frames: Number of frames (0 for unknown/streaming)

    Returns:
        WAV header bytes
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Data size: if unknown, advertise a maximal RIFF payload so streaming
    # clients can begin playback before the final frame count is known.
    data_size = num_frames * block_align
    if num_frames == 0:
        data_size = 0xFFFFFFFF - 36

    chunk_size = 36 + data_size

    header = io.BytesIO()
    header.write(b'RIFF')
    header.write(struct.pack('<I', chunk_size))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
    header.write(struct.pack('<H', 1))  # AudioFormat (1 for PCM)
    header.write(struct.pack('<H', num_channels))
    header.write(struct.pack('<I', sample_rate))
    header.write(struct.pack('<I', byte_rate))
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size))

    return header.getvalue()


def tensor_to_pcm_bytes(chunk_tensor: torch.Tensor) -> bytes:
    """
    Convert audio tensor chunk to 16-bit PCM bytes.

    Args:
        chunk_tensor: Audio tensor chunk

    Returns:
        PCM audio bytes
    """
    if chunk_tensor.is_cuda:
        chunk_tensor = chunk_tensor.cpu()

    if chunk_tensor.dim() == 1:
        chunk_tensor = chunk_tensor.unsqueeze(0)
    elif chunk_tensor.dim() != 2:
        raise ValueError(
            f'Expected chunk tensor with 1 or 2 dims for PCM conversion, got shape {tuple(chunk_tensor.shape)}'
        )

    # Convert to 16-bit PCM
    pcm = (chunk_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
    return pcm.numpy().tobytes()


def compute_rms_dbfs(audio_tensor: torch.Tensor) -> float:
    """Compute RMS loudness in dBFS for an audio tensor."""
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.dim() != 2:
        raise ValueError(
            f'Expected audio tensor with 1 or 2 dims for RMS calculation, got shape {tuple(audio_tensor.shape)}'
        )

    rms = audio_tensor.float().pow(2).mean().sqrt().item()
    return 20.0 * math.log10(rms + 1e-9)


def normalize_voice_wavs(
    voices_dir: str | None,
    skip: bool = False,
    target_dbfs: float = TARGET_VOICE_DBFS,
    tolerance_db: float = VOICE_NORMALIZE_TOLERANCE_DB,
) -> None:
    """
    Normalize loudness for WAV voices in-place before they are loaded by the model.

    If this is ever called after voice states have been cached, the voice cache
    should be cleared so model states are regenerated from the normalized files.
    """
    if skip:
        logger.info('Skipping startup voice WAV normalization.')
        return

    if not voices_dir:
        return

    voice_root = Path(voices_dir)
    if not voice_root.is_dir():
        logger.warning('Voice normalization skipped; voices directory not found: %s', voices_dir)
        return

    for wav_path in sorted(voice_root.glob('*.wav')):
        try:
            audio_tensor, sample_rate = torchaudio.load(str(wav_path))
        except (OSError, RuntimeError) as exc:
            logger.warning("Could not read voice WAV '%s': %s", wav_path, exc)
            continue

        if audio_tensor.numel() == 0:
            logger.warning("Voice WAV '%s' is empty; skipping normalization.", wav_path)
            continue

        bits_per_sample = _get_bits_per_sample(str(wav_path))
        if sample_rate != 24000:
            logger.warning(
                "Voice WAV '%s' has sample rate %s Hz; loudness normalization only, no resampling performed.",
                wav_path,
                sample_rate,
            )
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
            logger.warning(
                "Voice WAV '%s' has %s channels; loudness normalization only, no channel remix performed.",
                wav_path,
                audio_tensor.shape[0],
            )
        if bits_per_sample not in (None, 16):
            logger.warning(
                "Voice WAV '%s' reports %s-bit audio; loudness normalization only, no bit-depth conversion performed.",
                wav_path,
                bits_per_sample,
            )

        current_dbfs = compute_rms_dbfs(audio_tensor)
        if abs(current_dbfs - target_dbfs) <= tolerance_db:
            logger.debug("Voice WAV '%s' already within %.1f dB of target loudness.", wav_path, tolerance_db)
            continue

        gain = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
        normalized = (audio_tensor * gain).clamp(-1.0, 1.0)

        try:
            torchaudio.save(
                str(wav_path),
                normalized,
                sample_rate,
                format='wav',
                encoding='PCM_S',
                bits_per_sample=16,
            )
        except Exception as exc:
            logger.warning("Could not normalize voice WAV '%s': %s", wav_path, exc)
            continue

        logger.info(
            "Normalized voice WAV '%s' from %.2f dBFS to target %.2f dBFS.",
            wav_path,
            current_dbfs,
            target_dbfs,
        )


def _get_bits_per_sample(path: str) -> int | None:
    """Best-effort bit depth lookup for diagnostic logging."""
    try:
        info = sf.info(path)
    except Exception:
        return None
    return getattr(info, 'bits_per_sample', None)


def get_mime_type(fmt: str) -> str:
    """
    Get the MIME type for an audio format.

    Args:
        fmt: Audio format string

    Returns:
        MIME type string
    """
    mime_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'pcm': 'audio/L16',
        'opus': 'audio/opus',
        'aac': 'audio/aac',
        'flac': 'audio/flac',
    }
    return mime_types.get(fmt, f'audio/{fmt}')
