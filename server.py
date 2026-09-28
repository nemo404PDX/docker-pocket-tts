#!/usr/bin/env python3
"""
PocketTTS OpenAI-Compatible Server

A drop-in replacement for OpenAI's TTS API using the pocket-tts model.
Supports streaming, custom voices, and runs on CPU.

Usage:
    python server.py [OPTIONS]

    # Or with environment variables:
    POCKET_TTS_PORT=8080 python server.py
"""

import argparse
import os
import sys

from pathlib import Path


def _load_env_file() -> None:
    """Load .env key/value pairs into process env if python-dotenv is not used."""
    env_path = Path(__file__).resolve().parent / '.env'
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export ') :].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()

from app import create_app, init_tts_service
from app.config import Config
from app.logging_config import get_logger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PocketTTS OpenAI-Compatible Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with defaults
    python server.py

    # Custom port and voices directory
    python server.py --port 8080 --voices-dir ./my_voices

    # Enable streaming by default
    python server.py --stream

Environment Variables (full reference in README.md):
    POCKET_TTS_HOST         Server host (default: 0.0.0.0)
    POCKET_TTS_PORT         Server port (default: 49112)
    POCKET_TTS_THREADS      Waitress worker threads (default: 4)
    POCKET_TTS_MODEL_PATH   Path to model config file (.yaml) or variant name
    POCKET_TTS_VOICES_DIR   Path to voices directory
    POCKET_TTS_STREAM_DEFAULT Enable streaming by default
    POCKET_TTS_MAX_INPUT_CHARS Reject synthesis input longer than this (default: 32000; 0 disables)
    POCKET_TTS_API_KEY      Optional bearer token required on all routes except /health,/ready
    POCKET_TTS_SEGMENT_GAP_MS Silence inserted between segments in ms (default: 200; 0 disables)
    POCKET_TTS_MAX_CACHED_VOICES Bound for the loaded-voice LRU cache (default: 8; 0 unbounded)
    POCKET_TTS_TEXT_PREPROCESS_DEFAULT Enable text preprocessing by default
    POCKET_TTS_NORMALIZE_DEFAULT Enable normalization by default
    POCKET_TTS_SEGMENTATION_DEFAULT Default segmentation mode
    POCKET_TTS_SKIP_CODE_BLOCKS Replace fenced code blocks before synthesis
    POCKET_TTS_CODE_BLOCK_PHRASE Placeholder text for code blocks
    POCKET_TTS_EMOJI_HANDLING Emoji handling mode
    POCKET_TTS_BRIEFING_MODE Enable briefing-oriented preprocessing defaults
    POCKET_TTS_DROP_SOURCE_LINKS Drop trailing --- SOURCE LINKS --- section
    POCKET_TTS_SKIP_VOICE_NORMALIZE Skip startup voice WAV loudness normalization
    POCKET_TTS_LOG_DIR      Log directory path
        """,
    )

    parser.add_argument(
        '--host', type=str, default=Config.HOST, help=f'Host to bind to (default: {Config.HOST})'
    )
    parser.add_argument(
        '--port', type=int, default=Config.PORT, help=f'Port to listen on (default: {Config.PORT})'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=Config.MODEL_PATH,
        dest='model_path',
        help='Path to model config file (.yaml) or variant name',
    )
    parser.add_argument(
        '--voices-dir',
        type=str,
        default=Config.VOICES_DIR,
        dest='voices_dir',
        help='Directory containing voice files',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        default=Config.STREAM_DEFAULT,
        help='Enable streaming by default for all requests',
    )
    parser.add_argument(
        '--text-preprocess',
        action='store_true',
        default=None,
        help='Enable normalization for all requests (legacy alias for --normalize)',
    )
    parser.add_argument(
        '--normalize',
        action=argparse.BooleanOptionalAction,
        default=Config.TTS_NORMALIZE_DEFAULT,
        help='Enable normalization by default for all requests',
    )
    parser.add_argument(
        '--segmentation',
        type=str,
        default=Config.TTS_SEGMENTATION_DEFAULT,
        choices=['none', 'paragraph', 'sentence'],
        help=f"Default segmentation mode (default: {Config.TTS_SEGMENTATION_DEFAULT})",
    )
    parser.add_argument(
        '--skip-code-blocks',
        action=argparse.BooleanOptionalAction,
        default=Config.TTS_SKIP_CODE_BLOCKS,
        dest='skip_code_blocks',
        help='Replace fenced code blocks with a placeholder before synthesis',
    )
    parser.add_argument(
        '--code-block-phrase',
        type=str,
        default=Config.TTS_CODE_BLOCK_PHRASE,
        dest='code_block_phrase',
        help='Replacement phrase used when code blocks are skipped',
    )
    parser.add_argument(
        '--emoji-handling',
        type=str,
        default=Config.TTS_EMOJI_HANDLING,
        choices=['strip', 'describe', 'keep'],
        dest='emoji_handling',
        help=f"Emoji handling mode (default: {Config.TTS_EMOJI_HANDLING})",
    )
    parser.add_argument(
        '--briefing-mode',
        action=argparse.BooleanOptionalAction,
        default=Config.TTS_BRIEFING_MODE,
        dest='briefing_mode',
        help='Enable briefing profile defaults (normalize + sentence segmentation)',
    )
    parser.add_argument(
        '--drop-source-links',
        action=argparse.BooleanOptionalAction,
        default=Config.TTS_DROP_SOURCE_LINKS,
        dest='drop_source_links',
        help='Drop trailing --- SOURCE LINKS --- section before synthesis',
    )
    parser.add_argument(
        '--skip-voice-normalize',
        action=argparse.BooleanOptionalAction,
        default=Config.SKIP_VOICE_NORMALIZE,
        dest='skip_voice_normalize',
        help='Skip startup loudness normalization for custom WAV voice files',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=Config.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        dest='log_level',
        help='Logging level',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=Config.THREADS,
        dest='threads',
        help=f'Waitress worker threads (default: {Config.THREADS})',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    effective_normalize = args.normalize if args.text_preprocess is None else True

    setup_logging(args.log_level)

    # Create app
    app = create_app(
        {
            'STREAM_DEFAULT': args.stream,
            'TEXT_PREPROCESS_DEFAULT': effective_normalize,
            'TTS_NORMALIZE_DEFAULT': effective_normalize,
            'TTS_SEGMENTATION_DEFAULT': args.segmentation,
            'TTS_SKIP_CODE_BLOCKS': args.skip_code_blocks,
            'TTS_CODE_BLOCK_PHRASE': args.code_block_phrase,
            'TTS_EMOJI_HANDLING': args.emoji_handling,
            'TTS_BRIEFING_MODE': args.briefing_mode,
            'TTS_DROP_SOURCE_LINKS': args.drop_source_links,
            'SKIP_VOICE_NORMALIZE': args.skip_voice_normalize,
        }
    )

    logger = get_logger()

    # Initialize TTS service
    try:
        init_tts_service(
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            skip_voice_normalize=args.skip_voice_normalize,
        )
    except Exception as e:
        logger.error(f'Failed to initialize TTS service: {e}')
        sys.exit(1)

    # Start server with Waitress (production WSGI server)
    try:
        from waitress import serve

        logger.info(f'Starting PocketTTS server on http://{args.host}:{args.port}')
        logger.info('Press Ctrl+C to stop')

        serve(app, host=args.host, port=args.port, threads=args.threads, url_scheme='http')

    except ImportError:
        logger.warning('Waitress not installed, falling back to Flask dev server')
        logger.warning('Install waitress for production: pip install waitress')
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
