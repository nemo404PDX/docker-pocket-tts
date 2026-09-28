from pathlib import Path
import sys

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
