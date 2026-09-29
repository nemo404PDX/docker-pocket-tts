"""Tests for the HTTP layer hardening in app/api/routes.py.

Covers the P1 additions without loading the real model:
  - optional bearer-token auth (POCKET_TTS_API_KEY) and its /health,/ready exemption
  - input-length cap (POCKET_TTS_MAX_INPUT_CHARS) → 413
  - OpenAI-shaped error envelope ({'error': {'message', 'type', 'code'}})
  - inter-segment silence gap (POCKET_TTS_SEGMENT_GAP_MS) on the file path

The Flask app is built via create_app() with config overrides; routes that don't need
the model loaded are exercised directly, and _synthesize_segments/_silence_tensor are
unit-tested against a fake TTS service.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from app import create_app
from app.api import routes


def make_app(**overrides):
    return create_app(overrides or None)


# ─────────────────────────────────────────────
# Optional API key auth
# ─────────────────────────────────────────────
def test_no_auth_required_when_key_unset():
    client = make_app().test_client()
    # /v1/voices works without a model loaded (it only lists builtins + dir).
    resp = client.get('/v1/voices')
    assert resp.status_code == 200


def test_missing_key_is_rejected_when_set():
    client = make_app(API_KEY='secret').test_client()
    resp = client.get('/v1/voices')
    assert resp.status_code == 401
    body = resp.get_json()
    # OpenAI-shaped envelope.
    assert body['error']['type'] == 'authentication_error'


def test_correct_bearer_key_is_accepted():
    client = make_app(API_KEY='secret').test_client()
    resp = client.get('/v1/voices', headers={'Authorization': 'Bearer secret'})
    assert resp.status_code == 200


def test_wrong_key_is_rejected():
    client = make_app(API_KEY='secret').test_client()
    resp = client.get('/v1/voices', headers={'Authorization': 'Bearer nope'})
    assert resp.status_code == 401


def test_health_and_ready_are_exempt_from_auth():
    client = make_app(API_KEY='secret').test_client()
    # Model not loaded → these return 503, but crucially NOT 401.
    for path in ('/health', '/ready'):
        resp = client.get(path)
        assert resp.status_code != 401


# ─────────────────────────────────────────────
# /v1/voices shape (OpenAI-compatible created_at)
# ─────────────────────────────────────────────
def test_voices_entries_include_created_at():
    client = make_app().test_client()
    resp = client.get('/v1/voices')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['object'] == 'list'
    data = body['data']
    assert len(data) > 0
    for entry in data:
        # Every voice carries a stable integer created_at (OpenAI-compatible field).
        assert isinstance(entry['created_at'], int)
        if entry.get('type') == 'builtin':
            # Built-ins are part of the model → epoch 0.
            assert entry['created_at'] == 0


# ─────────────────────────────────────────────
# Input-length cap
# ─────────────────────────────────────────────
def test_oversized_input_is_rejected_with_413():
    client = make_app(MAX_INPUT_CHARS=100).test_client()
    resp = client.post('/tts', json={'input': 'x' * 200})
    assert resp.status_code == 413
    body = resp.get_json()
    assert body['error']['code'] == 'input_too_long'


def test_input_at_or_under_cap_is_not_rejected_by_length():
    client = make_app(MAX_INPUT_CHARS=100).test_client()
    # Exactly at the cap → passes the length check (fails later on model load, 500),
    # so it must NOT be a 413.
    resp = client.post('/tts', json={'input': 'x' * 100})
    assert resp.status_code != 413


def test_zero_cap_disables_the_limit():
    client = make_app(MAX_INPUT_CHARS=0).test_client()
    # Huge input is not rejected by length (fails later on model load, 500), so no 413.
    resp = client.post('/tts', json={'input': 'x' * 10_000})
    assert resp.status_code != 413


# ─────────────────────────────────────────────
# OpenAI-shaped error envelope
# ─────────────────────────────────────────────
def test_missing_body_returns_openai_envelope():
    client = make_app().test_client()
    resp = client.post('/tts', data='not json', content_type='text/plain')
    assert resp.status_code == 400
    body = resp.get_json()
    assert set(body['error']) >= {'message', 'type'}


def test_missing_input_returns_openai_envelope():
    client = make_app().test_client()
    # Non-empty dict with no 'input'/'text' → hits the missing-input branch (not
    # the falsy "missing JSON body" branch).
    resp = client.post('/tts', json={'voice': 'alba'})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error']['code'] == 'missing_input'


# ─────────────────────────────────────────────
# response_format handling
# ─────────────────────────────────────────────
def test_root_advertises_supported_formats():
    client = make_app().test_client()
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.get_json()
    # Core formats are always advertised.
    assert {'wav', 'mp3', 'flac', 'pcm'} <= set(body['supported_formats'])


def test_unsupported_format_returns_400(monkeypatch):
    # Force a known supported set (independent of whether this env has ffmpeg).
    monkeypatch.setattr(routes, 'is_supported_format', lambda fmt: fmt in {'wav', 'mp3', 'flac', 'pcm'})

    client = make_app().test_client()
    resp = client.post('/tts', json={'input': 'hi', 'response_format': 'aac'})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error']['code'] == 'unsupported_format'


def test_supported_format_passes_validation(monkeypatch):
    monkeypatch.setattr(routes, 'is_supported_format', lambda fmt: fmt in {'wav', 'mp3', 'flac', 'pcm'})

    client = make_app().test_client()
    # mp3 is supported → passes format validation (fails later on model load, 500),
    # so it must NOT be a 400 unsupported_format.
    resp = client.post('/tts', json={'input': 'hi', 'response_format': 'mp3'})
    assert resp.status_code != 400 or resp.get_json()['error'].get('code') != 'unsupported_format'


# ─────────────────────────────────────────────
# Inter-segment silence gap (file path)
# ─────────────────────────────────────────────
class _FakeTTS:
    sample_rate = 24000

    def generate_audio(self, voice_state, text):
        # A fixed 100-sample "segment" per call.
        return torch.full((1, 100), 0.5, dtype=torch.float32)


def test_silence_tensor_disabled_returns_none():
    assert routes._silence_tensor(_FakeTTS(), gap_ms=0) is None
    assert routes._silence_tensor(_FakeTTS(), gap_ms=-5) is None


def test_silence_tensor_length_matches_gap_ms():
    # 200 ms at 24 kHz → 4800 samples.
    silence = routes._silence_tensor(_FakeTTS(), gap_ms=200)
    assert silence.shape[-1] == 4800
    assert torch.all(silence == 0)


def test_synthesize_segments_sentence_boundary_gets_full_gap():
    app = make_app(SEGMENT_GAP_MS=200)
    with app.app_context():
        out = routes._synthesize_segments(_FakeTTS(), {}, ['First sentence.', 'Second one.'])
    # 100 (seg1) + 4800 (full gap, seg1 ends a sentence) + 100 (seg2) = 5000 samples.
    assert out.shape[-1] == 5000
    # The gap region is silence; the segment regions are not.
    assert torch.all(out[100:4900] == 0)
    assert torch.any(out[:100] != 0)


def test_synthesize_segments_mid_sentence_gets_small_seam():
    app = make_app(SEGMENT_GAP_MS=200, SEGMENT_SEAM_MS=30)
    with app.app_context():
        out = routes._synthesize_segments(_FakeTTS(), {}, ['your code', 'is delayed'])
    # 100 (seg1) + 720 (30ms seam — mid-sentence split, not a sentence boundary) + 100.
    assert out.shape[-1] == 920
    assert torch.all(out[100:820] == 0)


def test_synthesize_segments_seam_capped_at_full_gap():
    # A seam larger than the full gap is clamped to the full gap, so a sentence boundary
    # never has less silence than a mid-clause split.
    app = make_app(SEGMENT_GAP_MS=20, SEGMENT_SEAM_MS=30)
    with app.app_context():
        out = routes._synthesize_segments(_FakeTTS(), {}, ['your code', 'is delayed'])
    # 100 + 480 (20ms, clamped from 30) + 100.
    assert out.shape[-1] == 680


def test_synthesize_segments_no_gap_when_disabled():
    app = make_app(SEGMENT_GAP_MS=0)
    with app.app_context():
        out = routes._synthesize_segments(_FakeTTS(), {}, ['First sentence.', 'Second one.'])
    # No gap → exactly 200 samples, no silence inserted (seam collapses to 0 too).
    assert out.shape[-1] == 200


def test_boundary_gap_ms_sentence_final():
    app = make_app(SEGMENT_GAP_MS=200, SEGMENT_SEAM_MS=30)
    with app.app_context():
        assert routes._boundary_gap_ms('First sentence.') == 200
        assert routes._boundary_gap_ms('Really?') == 200
        # Trailing quote/bracket is ignored when detecting the boundary.
        assert routes._boundary_gap_ms('He said "stop."') == 200


def test_boundary_gap_ms_mid_sentence():
    app = make_app(SEGMENT_GAP_MS=200, SEGMENT_SEAM_MS=30)
    with app.app_context():
        assert routes._boundary_gap_ms('your code') == 30
        # Empty / whitespace-only segment → seam (no boundary).
        assert routes._boundary_gap_ms('   ') == 30


def test_synthesize_single_segment_has_no_gap():
    app = make_app(SEGMENT_GAP_MS=200)
    with app.app_context():
        out = routes._synthesize_segments(_FakeTTS(), {}, ['only one'])
    # A single segment is returned as-is (no leading/trailing silence).
    assert out.shape[-1] == 100
