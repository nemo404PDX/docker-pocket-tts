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


def test_keeps_contractions_by_default():
    # pocket-tts reads raw contractions natively (the tokenizer keeps them as clean tokens),
    # so expansion is off by default for natural, conversational delivery.
    assert normalize_tts_text("don't stop") == "don't stop"


def test_expands_contractions_when_enabled():
    config = TtsPipelineConfig(expand_contractions=True)
    assert normalize_tts_text("don't stop", config) == 'do not stop'
    assert normalize_tts_text("isn't it", config) == 'is not it'


def test_normalizes_curly_apostrophes():
    # Curly apostrophes are normalized to straight ones regardless of expansion.
    assert normalize_tts_text("It's fine") == "It's fine"


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
    # "OK" is a real English word, so it's read as a word (lowercased), not spelled out.
    assert segment_tts_text('OK. No.', config) == ['ok.', 'No.']


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
    # Contraction expansion is off by default, so "don't" survives intact.
    result = debug_normalize("Use **bold** and don't panic.")
    stages = [stage for stage, _output in result.steps]
    assert stages == [
        'sanitize',
        'emojiExpansion',
        'markdownCleanup',
        'numberNorm',
        'whitespaceCleanup',
    ]
    assert result.normalized == "Use bold and don't panic."


def test_debug_output_expands_contractions_when_enabled():
    config = TtsPipelineConfig(expand_contractions=True)
    result = debug_normalize("Use **bold** and don't panic.", config)
    assert result.normalized == 'Use bold and do not panic.'


def test_source_links_tail_is_removed_when_enabled():
    config = TtsPipelineConfig(drop_source_links_section=True)
    text = 'Story content.\n\n--- SOURCE LINKS ---\n- https://example.com'
    assert normalize_tts_text(text, config) == 'Story content.'


def test_source_links_tail_kept_when_disabled():
    config = TtsPipelineConfig(drop_source_links_section=False)
    text = 'Story content.\n\n--- SOURCE LINKS ---\n- https://example.com'
    normalized = normalize_tts_text(text, config)
    # The section is kept (only its all-caps words are lowercased for consistent reading).
    assert 'source links' in normalized


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
    # Single-uppercase words (ffmpeg, iPhone) tokenize cleanly and are left alone — only
    # tokens with an embedded acronym (two consecutive capitals after the first) decompose.
    assert normalize_tts_text('ffmpeg and iPhone') == 'ffmpeg and iPhone'


def test_mixed_case_embedded_acronym_is_decomposed():
    # "WebUI" shreds into letters under sentencepiece (▁We b U I); decomposing at case
    # boundaries gives a clean reading: real words lowercase, acronyms spelled out.
    assert normalize_tts_text('Open WebUI is nice.') == 'Open web U I is nice.'


def test_mixed_case_word_acronym_part_is_read_as_a_word():
    # The acronym part of "SwiftUI" spells out; the word part lowercases (CMU decides).
    import app.services.normalize as normalize_mod

    if normalize_mod._load_pronouncing() is None:
        pytest.skip('pronouncing not available in this environment')
    assert normalize_tts_text('SwiftUI feels good.') == 'swift U I feels good.'


def test_mixed_case_single_uppercase_stays_intact():
    # GitHub / TypeScript tokenize into real word pieces (▁G it H ub) — no embedded
    # acronym, so they are left untouched rather than decomposed.
    assert normalize_tts_text('GitHub and TypeScript') == 'GitHub and TypeScript'


def test_mixed_case_acronym_with_digits_keeps_everything():
    # Digits inside a mixed-case token must not be dropped ("HTML5" -> "H T M L five").
    out = normalize_tts_text('We use HTML5 now.')
    assert 'H T M L' in out and 'five' in out


def test_acronym_expansion_preserves_surrounding_numbers():
    # Acronyms expand before number expansion; the IP address digits are untouched.
    out = normalize_tts_text('The IP is 192.168.1.5.')
    assert 'I P' in out and '192.168.1.5' in out


def test_lowercase_acronym_is_not_expanded():
    # "ip" (lowercase) is a normal word here, not the acronym — left alone.
    assert normalize_tts_text('the ip address') == 'the ip address'


def test_all_caps_common_word_is_read_as_a_word():
    # Real English words in all caps must be lowercased and read as a word, not spelled
    # out letter by letter ("SLOW" -> "slow", not "S L O W").
    assert normalize_tts_text('The SLOW path is fine.') == 'The slow path is fine.'


def test_all_caps_common_word_is_lowercased_fully():
    # Real words are lowercased fully (capitalization carries no pronunciation info).
    assert normalize_tts_text('NEVER do that') == 'never do that'


def test_time_markers_are_spelled_out_not_read_as_words():
    # AM/PM (and AD/BC) ARE real English words in the dictionary, but ALL-CAPS usage is a
    # time/era marker and must be spelled out — "nine AM" -> "nine A M", not "nine am".
    assert normalize_tts_text('See you at nine AM.') == 'See you at nine A M.'
    assert normalize_tts_text('Call me after five PM.') == 'Call me after five P M.'


def test_lowercase_time_marker_is_left_alone():
    # Lowercase "am" is the verb — untouched (the expansion is case-sensitive).
    assert normalize_tts_text('I am fine') == 'I am fine'


def test_word_acronym_is_read_as_a_word():
    # Word-acronyms are pronounced as a single word even though they aren't in the general
    # dictionary ("NASA" -> "nasa", not "N A S A"). Decided via the CMU Pronouncing
    # Dictionary: an initialism's entry is its letters spelled out, a word-acronym's is a
    # real pronunciation. (Skipped if pronouncing is unavailable.)
    import app.services.normalize as normalize_mod

    if normalize_mod._load_pronouncing() is None:
        pytest.skip('pronouncing not available in this environment')
    assert normalize_tts_text('NASA launched a probe.') == 'nasa launched a probe.'


def test_initialism_is_spelled_out():
    # True initialisms whose CMU entry IS their letters spelled out are read letter by
    # letter ("FBI" -> "F B I", not "fbi"). (Skipped if pronouncing is unavailable.)
    import app.services.normalize as normalize_mod

    if normalize_mod._load_pronouncing() is None:
        pytest.skip('pronouncing not available in this environment')
    assert normalize_tts_text('The FBI issued a report.') == 'The F B I issued a report.'


def test_word_acronym_fallback_without_pronouncing(monkeypatch):
    # If pronouncing is unavailable, word-acronyms fall back to letter-spelling (the
    # pre-CMU behavior) instead of being lowercased.
    import app.services.normalize as normalize_mod

    monkeypatch.setattr(normalize_mod, '_load_pronouncing', lambda: None)
    assert normalize_tts_text('NASA launched a probe.') == 'N A S A launched a probe.'


def test_all_caps_fallback_without_dictionary(monkeypatch):
    # If both dictionaries are unavailable, unknown all-caps tokens fall back to
    # letter-spelling (the pre-dictionary behavior) instead of being lowercased.
    import app.services.normalize as normalize_mod

    monkeypatch.setattr(normalize_mod, '_load_spell_checker', lambda: None)
    monkeypatch.setattr(normalize_mod, '_load_pronouncing', lambda: None)
    assert normalize_tts_text('The SLOW path') == 'The S L O W path'


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
    # "nemo404PDX" resolves to a spoken form at protection time (not frozen whole —
    # sentencepiece shreds a frozen one into subword garbage): words lowercase, acronyms
    # spelled out, digits expanded: "nemo four hundred four P D X".
    out = normalize_tts_text('Repo nemo404PDX/docker-pocket-tts updated.')
    assert 'nemo' in out and 'four hundred four' in out and 'P D X' in out


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


# ─────────────────────────────────────────────
# SCREAMING_SNAKE_CASE identifiers (env vars / constants)
# ─────────────────────────────────────────────
def test_snake_case_identifier_is_resolved_per_part():
    # FALLBACK_UNKNOWN_VOICE must not be letter-spelled as one fused blob; each part is
    # resolved on its own (real words lowercase, acronyms spelled out).
    out = normalize_tts_text('Set FALLBACK_UNKNOWN_VOICE to true.')
    assert 'F A L L B A C K' not in out  # no letter-spelling of the fused blob
    assert 'fallback unknown voice' in out


def test_snake_case_with_acronym_part_spells_out():
    import app.services.normalize as normalize_mod

    if normalize_mod._load_pronouncing() is None:
        pytest.skip('pronouncing not available in this environment')
    # The TTS part of a SCREAMING_SNAKE_CASE identifier spells out; word parts lowercase.
    out = normalize_tts_text('POCKET_TTS_MAX_FRAMES set.')
    assert 'pocket' in out and 'T T S' in out


def test_snake_case_with_digits_keeps_them():
    # Digits inside a SCREAMING_SNAKE_CASE identifier survive for number expansion.
    out = normalize_tts_text('MAX_RETRIES_3 exceeded.')
    assert 'three' in out or '3' in out


def test_snake_case_eos_part_is_spelled_out():
    # EOS is an English word (the spellchecker knows it) but inside an identifier it must be
    # spelled out, not read as the word "eos" — whisper heard "iOS"/"Essex" from that.
    out = normalize_tts_text('FRAMES_AFTER_EOS=6 set.')
    assert 'E O S' in out


def test_assignment_equals_is_read_as_equals():
    # A raw "=" reaching sentencepiece is an unknown token and the model emits noise around
    # it (whisper heard "Essex"). Assignment/comparison equals read as "equals"; the value
    # still flows through number expansion.
    out = normalize_tts_text('FRAMES_AFTER_EOS=6 set.')
    assert 'equals' in out and 'six' in out


def test_double_equals_is_read_as_equals():
    out = normalize_tts_text('If a==b then retry.')
    assert 'a equals b' in out or ('equals' in out)


def test_mixed_alnum_model_name_resolves_to_spoken_form():
    # llama3 / gpt2 style tokens resolve at protection time (words lowercase, digits intact
    # for number expansion) instead of being frozen whole and shredded by sentencepiece.
    out = normalize_tts_text('We run llama3 locally.')
    assert 'llama' in out and ('three' in out or '3' in out)


def test_repo_path_slash_is_read_as_slash():
    # A raw "/" reaching sentencepiece is <0x2F> (unknown control token) — the model emits
    # noise around it. Slashes between words read as "slash"; both sides still resolve.
    out = normalize_tts_text('The repo is nemo404PDX/docker-pocket-tts.')
    assert 'slash' in out and 'Docker Pocket T T S' in out


def test_and_or_slash_is_read_as_slash():
    out = normalize_tts_text('Use the cache or/and the fallback.')
    assert 'or slash and' in out


def test_bare_url_stays_intact():
    # A bare URL is protected whole — no "slash" injected, digits untouched.
    out = normalize_tts_text('See https://example.com/path for docs.')
    assert 'https://example.com/path' in out
    assert 'slash' not in out


def test_unix_path_stays_intact():
    # Frozen unix paths keep their slashes (no "slash" injection).
    out = normalize_tts_text('The file is at /tmp/a/b.txt.')
    assert '/tmp/a/b.txt' in out
    assert 'slash' not in out


# ─────────────────────────────────────────────
# Product name (pocket-tts) spoken form
# ─────────────────────────────────────────────
def test_product_name_hyphenated_reads_as_pocket_tts():
    # "pocket-tts" is the product's own name — it reads as "Pocket T T S", NOT
    # "pocket text to speech" (the curated 'TTS' expansion) and not shredded.
    out = normalize_tts_text('This runs on pocket-tts.')
    assert 'Pocket T T S' in out
    assert 'text to speech' not in out


def test_product_name_camelcase_reads_as_pocket_tts():
    out = normalize_tts_text('PocketTTS is fast.')
    assert 'Pocket T T S' in out
    assert 'text to speech' not in out


def test_product_name_docker_prefixed_reads_as_docker_pocket_tts():
    out = normalize_tts_text('The docker-pocket-tts repo hosts it.')
    assert 'Docker Pocket T T S' in out
    assert 'text to speech' not in out


def test_product_name_in_path_is_spoken():
    # Inside a repo path the product name is still spoken — that IS its own name being
    # referenced ("nemo404PDX/docker-pocket-tts" -> ".../Docker Pocket T T S"). The docker
    # prefix must be consumed as one unit (no dangling "docker-"), and the mixed-alnum
    # username resolves to a spoken form rather than staying frozen.
    out = normalize_tts_text('Clone nemo404PDX/docker-pocket-tts now.')
    assert 'nemo' in out and 'four hundred four' in out and 'P D X' in out
    assert 'Docker Pocket T T S' in out
    assert 'text to speech' not in out


def test_product_name_longer_token_stays_filename():
    # A longer token that merely contains the name (pocket-tts-server) is NOT the product
    # name — it stays a filename, untouched.
    out = normalize_tts_text('The pocket-tts-server package loads.')
    assert 'pocket-tts-server' in out
    assert 'text to speech' not in out


# ─────────────────────────────────────────────
# Number + unit tokens (2000ms / 2000 ms) reach the unit expander
# ─────────────────────────────────────────────
def test_glued_number_unit_is_expanded():
    # "2000ms" used to be frozen by the bare-hash protection and reach TTS unexpanded;
    # now it reads as "two thousand milliseconds".
    out = normalize_tts_text('Latency was 2000ms.')
    assert 'two thousand milliseconds' in out


def test_spaced_number_unit_is_expanded():
    out = normalize_tts_text('Latency was 2000 ms.')
    assert 'two thousand milliseconds' in out


def test_glued_khz_is_expanded():
    # "24kHz" (hex-looking) must not be frozen as a hash; it reads as kilohertz.
    out = normalize_tts_text('Sample rate is 24kHz.')
    assert 'twenty-four kilohertz' in out


def test_bare_git_hash_still_protected_after_hex_fix():
    # The hex-only hash regex must still protect real git hashes (digits + a-f letters).
    out = normalize_tts_text('Commit 68187ab is pushed.')
    assert '68187ab' in out
    assert 'thousand' not in out


def test_non_hex_alnum_token_is_not_a_hash():
    # A token with non-hex letters (g-z) is NOT a git hash — it must not be frozen.
    out = normalize_tts_text('The word master stays.')
    assert 'master' in out


def test_plus_connector_glued_reads_as_plus():
    # "token+download" used to reach sentencepiece with the raw "+" (unknown token -> noise);
    # it now reads as "token plus download".
    out = normalize_tts_text('the token+download path was broken')
    assert 'token plus download' in out


def test_plus_connector_spaced_reads_as_plus():
    out = normalize_tts_text('use token + download for this')
    assert 'token plus download' in out


def test_plus_between_numbers_expands_both_sides():
    # The connector runs before number expansion, so "2+2" -> "two plus two".
    out = normalize_tts_text('so 2+2 is four.')
    assert 'two plus two' in out


def test_cpp_double_plus_is_left_alone():
    # C++ has no word char after the second "+", so it stays as-is (sentencepiece handles it).
    out = normalize_tts_text('C++ and Rust are languages')
    assert 'C++' in out
    assert 'plus' not in out


def test_plus_inside_protected_filename_is_untouched():
    # A "+" inside a protected filename token must not become "plus".
    out = normalize_tts_text('save it as my+file.txt now')
    assert 'my+file.txt' in out
