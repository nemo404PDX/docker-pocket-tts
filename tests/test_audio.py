from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import audio as audio_module


def test_tensor_to_pcm_bytes_rejects_3d_input():
    with pytest.raises(ValueError, match='Expected chunk tensor with 1 or 2 dims'):
        audio_module.tensor_to_pcm_bytes(torch.zeros((1, 1, 8)))


def test_compute_rms_dbfs_is_stable_for_known_level():
    tensor = torch.full((1, 16), 0.1, dtype=torch.float32)
    dbfs = audio_module.compute_rms_dbfs(tensor)
    assert dbfs == pytest.approx(-20.0, abs=0.1)


def test_normalize_voice_wavs_skips_when_requested(tmp_path, monkeypatch):
    wav_path = tmp_path / 'voice.wav'
    wav_path.write_bytes(b'RIFF')

    called = {'load': 0}

    def fake_load(_path):
        called['load'] += 1
        return torch.full((1, 8), 0.5), 24000

    monkeypatch.setattr(audio_module.torchaudio, 'load', fake_load)

    audio_module.normalize_voice_wavs(str(tmp_path), skip=True)

    assert called['load'] == 0


def test_normalize_voice_wavs_rewrites_loud_file(tmp_path, monkeypatch):
    wav_path = tmp_path / 'voice.wav'
    wav_path.write_bytes(b'RIFF')

    saved: list[tuple[str, torch.Tensor, int, str | None, str | None, int | None]] = []

    def fake_load(path):
        assert path == str(wav_path)
        return torch.full((1, 16), 0.5, dtype=torch.float32), 24000

    def fake_save(path, tensor, sample_rate, format=None, encoding=None, bits_per_sample=None):
        saved.append((path, tensor.clone(), sample_rate, format, encoding, bits_per_sample))

    def fake_info(_path):
        return type('Info', (), {'bits_per_sample': 16})()

    monkeypatch.setattr(audio_module.torchaudio, 'load', fake_load)
    monkeypatch.setattr(audio_module.torchaudio, 'save', fake_save)
    monkeypatch.setattr(audio_module.torchaudio, 'info', fake_info)

    audio_module.normalize_voice_wavs(str(tmp_path))

    assert len(saved) == 1
    saved_path, normalized_tensor, sample_rate, fmt, encoding, bits_per_sample = saved[0]
    assert saved_path == str(wav_path)
    assert sample_rate == 24000
    assert fmt == 'wav'
    assert encoding == 'PCM_S'
    assert bits_per_sample == 16
    assert audio_module.compute_rms_dbfs(normalized_tensor) == pytest.approx(
        audio_module.TARGET_VOICE_DBFS, abs=0.2
    )


def test_normalize_voice_wavs_leaves_in_range_file_alone(tmp_path, monkeypatch):
    wav_path = tmp_path / 'voice.wav'
    wav_path.write_bytes(b'RIFF')

    saved: list[tuple[str, torch.Tensor, int]] = []

    def fake_load(_path):
        return torch.full((1, 16), 0.1, dtype=torch.float32), 24000

    def fake_save(path, tensor, sample_rate, format=None, encoding=None, bits_per_sample=None):
        saved.append((path, tensor.clone(), sample_rate))

    def fake_info(_path):
        return type('Info', (), {'bits_per_sample': 16})()

    monkeypatch.setattr(audio_module.torchaudio, 'load', fake_load)
    monkeypatch.setattr(audio_module.torchaudio, 'save', fake_save)
    monkeypatch.setattr(audio_module.torchaudio, 'info', fake_info)

    audio_module.normalize_voice_wavs(str(tmp_path))

    assert saved == []


def test_normalize_voice_wavs_skips_corrupt_file(tmp_path, monkeypatch):
    wav_path = tmp_path / 'voice.wav'
    wav_path.write_bytes(b'RIFF')

    def fake_load(_path):
        raise RuntimeError('corrupt wav')

    monkeypatch.setattr(audio_module.torchaudio, 'load', fake_load)

    audio_module.normalize_voice_wavs(str(tmp_path))


def test_normalize_voice_wavs_warns_but_still_normalizes_non_24khz(tmp_path, monkeypatch):
    wav_path = tmp_path / 'voice.wav'
    wav_path.write_bytes(b'RIFF')

    saved: list[tuple[str, torch.Tensor, int, str | None, str | None, int | None]] = []

    def fake_load(_path):
        return torch.full((2, 16), 0.5, dtype=torch.float32), 48000

    def fake_save(path, tensor, sample_rate, format=None, encoding=None, bits_per_sample=None):
        saved.append((path, tensor.clone(), sample_rate, format, encoding, bits_per_sample))

    def fake_info(_path):
        return type('Info', (), {'bits_per_sample': 24})()

    monkeypatch.setattr(audio_module.torchaudio, 'load', fake_load)
    monkeypatch.setattr(audio_module.torchaudio, 'save', fake_save)
    monkeypatch.setattr(audio_module.torchaudio, 'info', fake_info)

    audio_module.normalize_voice_wavs(str(tmp_path))

    assert len(saved) == 1
    assert saved[0][2] == 48000
    assert saved[0][3] == 'wav'
    assert saved[0][4] == 'PCM_S'
    assert saved[0][5] == 16
