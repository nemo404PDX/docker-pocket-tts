from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.normalize import (
    TtsPipelineConfig,
    debug_normalize,
    normalize_tts_text,
    segment_tts_text,
)


def test_expands_safe_contractions():
    assert normalize_tts_text("don't stop") == 'do not stop'


def test_normalizes_curly_apostrophes():
    assert normalize_tts_text("It's fine") == 'It is fine'


def test_strips_basic_markdown():
    assert normalize_tts_text('Use **bold** and _italic_') == 'Use bold and italic'


def test_strips_html_tags():
    assert normalize_tts_text('Use <b>bold</b> and <em>italics</em>') == 'Use bold and italics'


def test_replaces_code_blocks_with_phrase():
    assert (
        normalize_tts_text('''```\ncode here\n```''')
        == ', see the code in the conversation history,'
    )


def test_expands_safe_abbreviations():
    assert normalize_tts_text('See e.g., the docs') == 'See for example the docs'


def test_leaves_bare_bc_unchanged():
    assert normalize_tts_text('bc it works') == 'bc it works'


def test_preserves_filename_tokens():
    assert normalize_tts_text('because.exe failed') == 'because.exe failed'
    assert normalize_tts_text('README.md') == 'README.md'


def test_normalizes_dot_prefixed_tokens():
    assert normalize_tts_text('load .env.local') == 'load dot env local'


def test_preserves_version_tokens():
    assert normalize_tts_text('v2.0.1 released') == 'v2.0.1 released'


def test_strips_emoji_and_collapses_whitespace_by_default():
    assert normalize_tts_text('text 😀 more') == 'text more'


def test_keeps_emoji_when_requested():
    config = TtsPipelineConfig(emoji_handling='keep')
    assert normalize_tts_text('text 😀 more', config) == 'text 😀 more'


def test_segments_none_mode():
    config = TtsPipelineConfig(segmentation='none')
    assert segment_tts_text('Hello world.', config) == ['Hello world.']


def test_segments_paragraph_mode():
    config = TtsPipelineConfig(segmentation='paragraph')
    assert segment_tts_text('Para one.\n\nPara two.', config) == ['Para one.', 'Para two.']


def test_segments_paragraph_mode_without_normalization():
    config = TtsPipelineConfig(normalize=False, segmentation='paragraph')
    assert segment_tts_text('Para one.\n\nPara two.', config) == ['Para one.', 'Para two.']


def test_segments_sentence_mode():
    config = TtsPipelineConfig(segmentation='sentence')
    assert segment_tts_text('First sentence. Second sentence.', config) == [
        'First sentence.',
        'Second sentence.',
    ]


def test_segments_sentence_mode_without_normalization():
    config = TtsPipelineConfig(normalize=False, segmentation='sentence')
    assert segment_tts_text('First sentence. Second sentence.', config) == [
        'First sentence.',
        'Second sentence.',
    ]


def test_sentence_mode_without_normalization_does_not_split_filenames():
    config = TtsPipelineConfig(normalize=False, segmentation='sentence')
    assert segment_tts_text('Load README.md now.', config) == ['Load README.md now.']


def test_sentence_mode_does_not_split_filenames():
    config = TtsPipelineConfig(segmentation='sentence')
    assert segment_tts_text('Load README.md now.', config) == ['Load README.md now.']


def test_sentence_mode_handles_dot_tokens_without_false_split():
    config = TtsPipelineConfig(segmentation='sentence')
    assert segment_tts_text('Load .env.local here.', config) == ['Load dot env local here.']


def test_short_sentences_are_preserved():
    config = TtsPipelineConfig(segmentation='sentence')
    assert segment_tts_text('OK. No.', config) == ['OK.', 'No.']


def test_long_segment_splits_at_word_boundary():
    config = TtsPipelineConfig(segmentation='sentence', max_segment_chars=40)
    text = (
        'This is a deliberately long sentence that should split at a word boundary '
        'without losing any text or punctuation.'
    )
    segments = segment_tts_text(text, config)
    assert len(segments) > 1
    assert all(len(segment) <= 40 for segment in segments[:-1])
    assert ' '.join(segments).replace('  ', ' ') == text


def test_long_segment_prefers_comma_break():
    # A comma inside the window is a natural pause point — split there instead of
    # mid-clause, mirroring pocket-tts's own fallback split set (`,;:`).
    config = TtsPipelineConfig(segmentation='sentence', normalize=False, max_segment_chars=40)
    text = 'Alpha beta gamma delta epsilon zeta, eta theta iota kappa lambda mu nu xi omicron tail end.'
    segments = segment_tts_text(text, config)
    assert segments[0] == 'Alpha beta gamma delta epsilon zeta,'
    assert all(len(segment) <= 40 for segment in segments[:-1])
    # No text is lost across the split.
    assert ' '.join(segments).replace('  ', ' ') == text


def test_long_segment_falls_back_to_word_boundary_without_comma():
    # No weak punctuation in the window → fall back to the last word boundary
    # before max_chars (previous behavior).
    config = TtsPipelineConfig(segmentation='sentence', normalize=False, max_segment_chars=40)
    text = 'One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen tail end.'
    segments = segment_tts_text(text, config)
    assert all(len(segment) <= 40 for segment in segments[:-1])
    assert ' '.join(segments).replace('  ', ' ') == text


def test_default_cap_keeps_segments_token_safe():
    # pocket-tts caps each generation chunk at 50 tokens; prose runs ~2.8-3.2
    # chars/token, so the default cap must keep comma-free sentences well under it.
    config = TtsPipelineConfig(segmentation='sentence', normalize=False)
    words = (
        'the quick brown fox jumps over the lazy dog and keeps running through the '
        'dark forest until it finally reaches the river bank at dawn'
    ).split()
    text = (' '.join(words * 3)).capitalize() + '.'
    segments = segment_tts_text(text, config)
    assert len(segments) > 1
    assert all(len(segment) <= 130 for segment in segments)
    assert ' '.join(segments).replace('  ', ' ') == text


def test_debug_output_contains_stage_trace():
    result = debug_normalize("Use **bold** and don't panic.")
    stages = [stage for stage, _output in result.steps]
    assert stages == [
        'sanitize',
        'emojiExpansion',
        'markdownCleanup',
        'numberNorm',
        'whitespaceCleanup',
    ]
    assert result.normalized == 'Use bold and do not panic.'


def test_source_links_tail_is_removed_when_enabled():
    config = TtsPipelineConfig(drop_source_links_section=True)
    text = 'Story content.\n\n--- SOURCE LINKS ---\n- https://example.com'
    assert normalize_tts_text(text, config) == 'Story content.'


def test_source_links_tail_kept_when_disabled():
    config = TtsPipelineConfig(drop_source_links_section=False)
    text = 'Story content.\n\n--- SOURCE LINKS ---\n- https://example.com'
    normalized = normalize_tts_text(text, config)
    assert 'SOURCE LINKS' in normalized


# ─────────────────────────────────────────────
# Client-marker stripping (configurable pattern)
# ─────────────────────────────────────────────
def test_default_pattern_strips_hermes_markers():
    text = 'Before.\n[terminal] some tool output\nAfter.'
    # The marker line is removed; the surrounding single newline is preserved.
    assert normalize_tts_text(text, TtsPipelineConfig(normalize=False)) == 'Before.\nAfter.'


def test_custom_marker_pattern_is_used():
    # A different client's marker token — the whole line containing it is removed.
    config = TtsPipelineConfig(normalize=False, marker_pattern=r'\[TOOL\]')
    text = 'Keep this.\n[TOOL] output here\nKeep that too.'
    assert normalize_tts_text(text, config) == 'Keep this.\nKeep that too.'


def test_custom_marker_pattern_with_bracket_terminator():
    # Bracket-terminated markers (like the Hermes default) stop at the closing bracket.
    config = TtsPipelineConfig(normalize=False, marker_pattern=r'\[OUT\][^\]]*\]')
    text = 'One.\n[OUT] payload [nested] tail\nTwo.'
    assert normalize_tts_text(text, config) == 'One.\nTwo.'


def test_empty_marker_pattern_disables_stripping():
    # Explicit empty string disables marker stripping (default pattern would remove the line).
    config = TtsPipelineConfig(normalize=False, marker_pattern='')
    text = 'Before.\n[terminal] tool output\nAfter.'
    assert '[terminal]' in normalize_tts_text(text, config)


def test_invalid_marker_pattern_is_ignored():
    # Invalid regex → warning + no stripping (must not raise).
    config = TtsPipelineConfig(normalize=False, marker_pattern='(unclosed')
    text = 'Before.\n[terminal] tool output\nAfter.'
    assert '[terminal]' in normalize_tts_text(text, config)


def test_none_marker_pattern_uses_default():
    # None (unset) behaves exactly like the default pattern.
    text = 'Before.\n[OUT-OF-BAND USER MESSAGE] x\nAfter.'
    assert normalize_tts_text(text, TtsPipelineConfig(normalize=False)) == 'Before.\nAfter.'


# ─────────────────────────────────────────────
# Acronym / tech-term expansion (consistent pronunciation)
# ─────────────────────────────────────────────
def test_expands_all_caps_acronyms_to_letters():
    assert normalize_tts_text('The IP and CPU are busy.') == 'The I P and C P U are busy.'


def test_expands_longer_terms_before_shorter():
    # HTTPS must not be partially eaten by HTTP.
    assert normalize_tts_text('Use HTTPS not HTTP.') == 'Use H T T P S not H T T P.'


def test_expands_acronyms_to_spoken_phrases():
    assert normalize_tts_text('The LLM uses NLP.') == 'The large language model uses natural language processing.'


def test_leaves_mixed_case_words_alone():
    # Case-sensitive: only all-caps tokens are expanded, so ffmpeg / iPhone stay intact.
    assert normalize_tts_text('ffmpeg and iPhone') == 'ffmpeg and iPhone'


def test_acronym_expansion_preserves_surrounding_numbers():
    # Acronyms expand before number expansion; the IP address digits are untouched.
    out = normalize_tts_text('The IP is 192.168.1.5.')
    assert 'I P' in out and '192.168.1.5' in out


def test_lowercase_acronym_is_not_expanded():
    # "ip" (lowercase) is a normal word here, not the acronym — left alone.
    assert normalize_tts_text('the ip address') == 'the ip address'


# ─────────────────────────────────────────────
# Hash / mixed-alnum protection (clients strip markdown before TTS)
# ─────────────────────────────────────────────
def test_bare_git_hash_is_not_number_expanded():
    # Open WebUI strips backticks, so hashes arrive bare. They must survive number
    # expansion intact instead of becoming "sixty-eight thousand one hundred eighty-sevenab".
    out = normalize_tts_text('Commit 68187ab is pushed.')
    assert '68187ab' in out
    assert 'thousand' not in out


def test_bare_hash_range_is_not_number_expanded():
    # Each half of a range like "52c36f7..68187ab" must stay intact.
    out = normalize_tts_text('Range 52c36f7..68187ab pushed.')
    assert '52c36f7' in out and '68187ab' in out


def test_mixed_alnum_username_is_not_number_expanded():
    # "nemo404PDX" must not become "nemo four hundred four P D X".
    out = normalize_tts_text('Repo nemo404PDX/docker-pocket-tts updated.')
    assert 'nemo404PDX' in out


def test_pure_numbers_still_expand():
    # The hash protection must not swallow ordinary numbers.
    out = normalize_tts_text('There are 12345 files.')
    assert 'twelve thousand three hundred forty-five' in out


def test_multiple_protected_tokens_restore_cleanly():
    # Regression: two protected tokens in one text used to lose their separator via the
    # italic-underscore markdown rule, leaking literal \x02PROT..\x03 garbage into TTS.
    out = normalize_tts_text('Commit `68187ab` and range 52c36f7..68187ab pushed.')
    assert '\x02' not in out and 'PROT' not in out
    assert '68187ab' in out and '52c36f7' in out


def test_token_cap_splits_hash_dense_segment():
    # Hash-dense text can exceed pocket-tts's 50-token chunk cap well under the character
    # limit; the tokenizer pass must split it. (Skipped if the model tokenizer is absent.)
    from app.services.normalize import _count_tokens

    if _count_tokens('hello world') is None:
        pytest.skip('sentencepiece tokenizer not available in this environment')

    config = TtsPipelineConfig(segmentation='sentence', normalize=False, max_segment_chars=200)
    text = ' '.join(['68187ab'] * 30) + '.'
    segments = segment_tts_text(text, config)
    assert len(segments) > 1
    for segment in segments:
        assert _count_tokens(segment) <= 50


def test_token_cap_disabled_keeps_single_segment():
    from app.services.normalize import _count_tokens

    if _count_tokens('hello world') is None:
        pytest.skip('sentencepiece tokenizer not available in this environment')

    # Char cap high enough that only the token pass could split it.
    config = TtsPipelineConfig(
        segmentation='sentence', normalize=False, max_segment_chars=400, max_segment_tokens=None
    )
    text = ' '.join(['68187ab'] * 30) + '.'
    segments = segment_tts_text(text, config)
    assert len(segments) == 1


def test_token_cap_enabled_splits_same_text():
    from app.services.normalize import _count_tokens

    if _count_tokens('hello world') is None:
        pytest.skip('sentencepiece tokenizer not available in this environment')

    # Same text, same char cap — only the token pass differs. It must split a segment
    # that fits comfortably under 400 chars but exceeds 50 tokens.
    config = TtsPipelineConfig(
        segmentation='sentence', normalize=False, max_segment_chars=400, max_segment_tokens=50
    )
    text = ' '.join(['68187ab'] * 30) + '.'
    segments = segment_tts_text(text, config)
    assert len(segments) > 1
    for segment in segments:
        assert _count_tokens(segment) <= 50
