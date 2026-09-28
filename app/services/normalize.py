"""
TTS-specific normalization and segmentation helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.services.preprocess import TextPreprocessor

StageName = Literal[
    'sanitize',
    'emojiExpansion',
    'markdownCleanup',
    'numberNorm',
    'whitespaceCleanup',
]

EmojiHandling = Literal['strip', 'describe', 'keep']
SegmentationMode = Literal['none', 'paragraph', 'sentence']

DEFAULT_CODE_BLOCK_PHRASE = ', see the code in the conversation history,'
MAX_SEGMENT_CHARS = 280
PATH_DOT_TOKEN = '\x02PD\x03'

_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
_INLINE_CODE_RE = re.compile(r'`([^`]+)`')
_IMAGE_RE = re.compile(r'!\[.*?\]\(.*?\)')
_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]*\)')
_HEADING_RE = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_HR_RE = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r'\*{3}([^*]+)\*{3}')
_BOLD_RE = re.compile(r'\*{2}([^*]+)\*{2}')
_ITALIC_STAR_RE = re.compile(r'\*([^*]+)\*')
_BOLD_UNDERSCORE_RE = re.compile(r'_{2}([^_]+)_{2}')
_ITALIC_UNDERSCORE_RE = re.compile(r'_([^_]+)_')
_BLOCKQUOTE_RE = re.compile(r'^>\s+', re.MULTILINE)
_UNORDERED_LIST_RE = re.compile(r'^[-*+]\s+', re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r'^\d+\.\s+', re.MULTILINE)
# Multiline tables: each row on its own line (starts with |, contains |)
_TABLE_MULTILINE_RE = re.compile(r'^(?:\|.*\|\s*\n?)+', re.MULTILINE)

# Single-line tables: all rows concatenated without newlines
# Pattern: | col | col | |---|---| | data | data | (multiple pipe-delimited segments in one line)
_TABLE_SINGLELINE_RE = re.compile(r'\|[^\n]*\|\s*\|[^\n]*\|\s*(?:\|[^\n]*\|\s*)+')
# Hermes-specific markers that leak into TTS text (opening/closing tags + tool output)
_HERMES_MARKER_RE = re.compile(
    r'/?\[[/\s]*(?:terminal|OUT-OF-BAND USER MESSAGE)[^\]]*\]',
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(
    r'(?<!\.)\b[a-z0-9_-]+\.[a-z0-9_-]{2,}(?:\.[a-z0-9._-]+)*\b', re.IGNORECASE
)
_VERSION_RE = re.compile(r'\bv\d+\.\d+(?:[\w.]+)?\b', re.IGNORECASE)
_UNIX_PATH_RE = re.compile(r'(?<!\w)/(?:[a-z0-9._-]+/)*[a-z0-9._-]+', re.IGNORECASE)
_INLINE_WHITESPACE_RE = re.compile(r'[^\S\n]{2,}')
_NEWLINE_PADDING_RE = re.compile(r'[^\S\n]*\n[^\S\n]*')
_SENTENCE_RE = re.compile(r'[^.!?]+[.!?]+(?:["\')\]]+)?|[^.!?]+$')
_SOURCE_LINKS_TAIL_RE = re.compile(r'(?im)^\s*---\s*SOURCE LINKS\s*---\s*$[\s\S]*$')
_EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001F5FF'
    '\U0001F600-\U0001F64F'
    '\U0001F680-\U0001F6FF'
    '\U0001F700-\U0001F77F'
    '\U0001F780-\U0001F7FF'
    '\U0001F800-\U0001F8FF'
    '\U0001F900-\U0001F9FF'
    '\U0001FA00-\U0001FA6F'
    '\U0001FA70-\U0001FAFF'
    '\u2600-\u26FF'
    '\u2700-\u27BF'
    ']+',
    re.UNICODE,
)

CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcan't\b", re.IGNORECASE), 'cannot'),
    (re.compile(r"\bwon't\b", re.IGNORECASE), 'will not'),
    (re.compile(r"\bdon't\b", re.IGNORECASE), 'do not'),
    (re.compile(r"\bdoesn't\b", re.IGNORECASE), 'does not'),
    (re.compile(r"\bdidn't\b", re.IGNORECASE), 'did not'),
    (re.compile(r"\bwouldn't\b", re.IGNORECASE), 'would not'),
    (re.compile(r"\bcouldn't\b", re.IGNORECASE), 'could not'),
    (re.compile(r"\bshouldn't\b", re.IGNORECASE), 'should not'),
    (re.compile(r"\bisn't\b", re.IGNORECASE), 'is not'),
    (re.compile(r"\baren't\b", re.IGNORECASE), 'are not'),
    (re.compile(r"\bwasn't\b", re.IGNORECASE), 'was not'),
    (re.compile(r"\bweren't\b", re.IGNORECASE), 'were not'),
    (re.compile(r"\bhaven't\b", re.IGNORECASE), 'have not'),
    (re.compile(r"\bhasn't\b", re.IGNORECASE), 'has not'),
    (re.compile(r"\bhadn't\b", re.IGNORECASE), 'had not'),
    (re.compile(r"\bmustn't\b", re.IGNORECASE), 'must not'),
    (re.compile(r"\bneedn't\b", re.IGNORECASE), 'need not'),
    (re.compile(r"\bi'm\b", re.IGNORECASE), 'I am'),
    (re.compile(r"\bi've\b", re.IGNORECASE), 'I have'),
    (re.compile(r"\bi'll\b", re.IGNORECASE), 'I will'),
    (re.compile(r"\bi'd\b", re.IGNORECASE), 'I would'),
    (re.compile(r"\byou're\b", re.IGNORECASE), 'you are'),
    (re.compile(r"\byou've\b", re.IGNORECASE), 'you have'),
    (re.compile(r"\byou'll\b", re.IGNORECASE), 'you will'),
    (re.compile(r"\byou'd\b", re.IGNORECASE), 'you would'),
    (re.compile(r"\bhe's\b", re.IGNORECASE), 'he is'),
    (re.compile(r"\bshe's\b", re.IGNORECASE), 'she is'),
    (re.compile(r"\bit's\b", re.IGNORECASE), 'it is'),
    (re.compile(r"\bit'll\b", re.IGNORECASE), 'it will'),
    (re.compile(r"\bwe're\b", re.IGNORECASE), 'we are'),
    (re.compile(r"\bwe've\b", re.IGNORECASE), 'we have'),
    (re.compile(r"\bwe'll\b", re.IGNORECASE), 'we will'),
    (re.compile(r"\bwe'd\b", re.IGNORECASE), 'we would'),
    (re.compile(r"\bthey're\b", re.IGNORECASE), 'they are'),
    (re.compile(r"\bthey've\b", re.IGNORECASE), 'they have'),
    (re.compile(r"\bthey'll\b", re.IGNORECASE), 'they will'),
    (re.compile(r"\bthey'd\b", re.IGNORECASE), 'they would'),
    (re.compile(r"\bhe'd\b", re.IGNORECASE), 'he would'),
    (re.compile(r"\bshe'd\b", re.IGNORECASE), 'she would'),
    (re.compile(r"\bhe'll\b", re.IGNORECASE), 'he will'),
    (re.compile(r"\bshe'll\b", re.IGNORECASE), 'she will'),
    (re.compile(r"\bthat's\b", re.IGNORECASE), 'that is'),
    (re.compile(r"\bthat'll\b", re.IGNORECASE), 'that will'),
    (re.compile(r"\bthere's\b", re.IGNORECASE), 'there is'),
    (re.compile(r"\bhere's\b", re.IGNORECASE), 'here is'),
    (re.compile(r"\bwhat's\b", re.IGNORECASE), 'what is'),
    (re.compile(r"\bwho's\b", re.IGNORECASE), 'who is'),
    (re.compile(r"\blet's\b", re.IGNORECASE), 'let us'),
)

ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\be\.g\.,?', re.IGNORECASE), 'for example'),
    (re.compile(r'\bi\.e\.,?', re.IGNORECASE), 'that is'),
    (re.compile(r'\betc\.', re.IGNORECASE), 'et cetera'),
    (re.compile(r'\bvs\.', re.IGNORECASE), 'versus'),
    (re.compile(r'\bw/', re.IGNORECASE), 'with'),
    (re.compile(r'\bN/A\b', re.IGNORECASE), 'not applicable'),
    (re.compile(r'\bTL;DR\b', re.IGNORECASE), "too long, did not read"),
)

PUNCT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r'\.{2,}'), '\x02EL\x03', ','),
    (re.compile(r'[—–]'), '\x02DA\x03', ','),
    (re.compile(r'[!?]{2,}'), '\x02MP\x03', '.'),
)

EMOJI_DESCRIPTIONS: dict[str, str] = {
    '😀': '<grin>',
    '😃': '<smile>',
    '😄': '<laugh>',
    '😁': '<grin>',
    '😆': '<laugh>',
    '😅': '<sweat laugh>',
    '😂': '<laugh>',
    '🤣': '<laugh>',
    '😊': '<smile>',
    '😇': '<smile>',
    '🙂': '<smile>',
    '😉': '<wink>',
    '😍': '<heart eyes>',
    '🥰': '<heart>',
    '😘': '<kiss>',
    '😜': '<tongue wink>',
    '😎': '<cool>',
    '😢': '<sad>',
    '😭': '<cry>',
    '😤': '<huff>',
    '😡': '<angry>',
    '🤔': '<thinking>',
    '🤷': '<shrug>',
    '👍': '<thumbs up>',
    '👎': '<thumbs down>',
    '👋': '<wave>',
    '🙏': '<please>',
    '💪': '<strong>',
    '🔥': '<fire>',
    '✅': '<check>',
    '❌': '<cross>',
    '⚠': '<warning>',
    '⚠️': '<warning>',
    '💡': '<idea>',
    '❤️': '<heart>',
    '💔': '<broken heart>',
    '⭐': '<star>',
    '🌟': '<star>',
    '🎉': '<celebrate>',
    '🎊': '<celebrate>',
    '🚀': '<rocket>',
    '💯': '<hundred>',
    '🤝': '<handshake>',
    '🫡': '<salute>',
}


@dataclass(slots=True)
class TtsPipelineConfig:
    normalize: bool = True
    skip_code_blocks: bool = True
    code_block_phrase: str = DEFAULT_CODE_BLOCK_PHRASE
    emoji_handling: EmojiHandling = 'strip'
    segmentation: SegmentationMode = 'sentence'
    max_segment_chars: int = MAX_SEGMENT_CHARS
    run_number_expansion: bool = True
    drop_source_links_section: bool = False


@dataclass(slots=True)
class TtsNormDebugResult:
    original: str
    normalized: str
    segments: list[str]
    steps: list[tuple[StageName, str]] = field(default_factory=list)


_NUMBER_PREPROCESSOR = TextPreprocessor(
    lowercase=False,
    replace_numbers=True,
    replace_floats=True,
    expand_newlines=False,
    expand_tilde=False,
    expand_abbreviations=False,
    expand_symbols=False,
    expand_contractions=False,
    expand_model_names=True,
    expand_ordinals=True,
    expand_percentages=True,
    expand_currency=True,
    expand_time=True,
    expand_ranges=True,
    expand_units=True,
    expand_scale_suffixes=True,
    expand_scientific_notation=True,
    expand_fractions=True,
    expand_decades=True,
    expand_phone_numbers=True,
    expand_ip_addresses=True,
    normalize_leading_decimals=True,
    expand_roman_numerals=False,
    remove_urls=False,
    remove_emails=False,
    remove_html=True,
    remove_hashtags=False,
    remove_mentions=False,
    remove_punctuation=False,
    remove_stopwords=False,
    normalize_unicode=True,
    remove_accents=False,
    remove_extra_whitespace=False,
)


def normalize_tts_text(text: str, config: TtsPipelineConfig | None = None) -> str:
    """Normalize text for TTS without segmenting it."""
    active_config = config or TtsPipelineConfig()
    normalized, _steps = _normalize_internal(text, active_config)
    return _restore_punctuation(normalized)


def segment_tts_text(text: str, config: TtsPipelineConfig | None = None) -> list[str]:
    """Normalize and segment text for TTS synthesis."""
    active_config = config or TtsPipelineConfig()
    normalized, _steps = _normalize_internal(text, active_config)
    return _segment_normalized_text(normalized, active_config)


def debug_normalize(text: str, config: TtsPipelineConfig | None = None) -> TtsNormDebugResult:
    """Return normalized text, final segments, and per-stage outputs."""
    active_config = config or TtsPipelineConfig()
    normalized, steps = _normalize_internal(text, active_config)
    return TtsNormDebugResult(
        original=text,
        normalized=_restore_punctuation(normalized),
        segments=_segment_normalized_text(normalized, active_config),
        steps=steps,
    )


def _normalize_internal(
    text: str, config: TtsPipelineConfig
) -> tuple[str, list[tuple[StageName, str]]]:
    steps: list[tuple[StageName, str]] = []

    result = _sanitize_text(text)
    if config.drop_source_links_section:
        result = strip_source_links_tail(result)
    # Strip tables early — before Hermes markers, since single-line tables may contain
    # marker text inside cells that would otherwise eat the entire line
    result = _strip_tables_for_tts(result)
    # Strip Hermes marker lines after tables so [terminal] inside table cells is already gone
    result = _strip_hermes_marker_lines(result)
    steps.append(('sanitize', result))

    if not config.normalize:
        result = _collapse_whitespace(result)
        steps.append(('emojiExpansion', result))
        steps.append(('markdownCleanup', result))
        steps.append(('numberNorm', result))
        steps.append(('whitespaceCleanup', result))
        return result, steps

    result = _apply_emoji_expansion(result, config.emoji_handling)
    steps.append(('emojiExpansion', result))

    result = _replace_structured_blocks_for_tts(result, config)
    protected_text, token_map = _protect_tokens(result)
    protected_text = _strip_markdown_syntax(protected_text)
    protected_text = _apply_emoji_strip(protected_text, config.emoji_handling)
    steps.append(('markdownCleanup', protected_text))

    protected_text = _normalize_apostrophes(protected_text)
    protected_text = _expand_contractions(protected_text)
    protected_text = _expand_abbreviations(protected_text)
    protected_text = _normalize_dot_prefixed_tokens(protected_text)
    if config.run_number_expansion:
        protected_text = _NUMBER_PREPROCESSOR.process(protected_text)
    protected_text = _protect_punctuation(protected_text)
    protected_text = _restore_tokens(protected_text, token_map)
    steps.append(('numberNorm', protected_text))

    protected_text = _collapse_whitespace(protected_text)
    steps.append(('whitespaceCleanup', protected_text))

    return protected_text, steps


def _sanitize_text(text: str) -> str:
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n')
    return cleaned


def _strip_hermes_marker_lines(text: str) -> str:
    """Remove entire lines that contain Hermes markers like [terminal] or [OUT-OF-BAND]."""
    # Match a full line (start to newline) containing any Hermes marker tag.
    # The trailing .* is greedy so it captures everything after the marker on the same line.
    return re.sub(
        r'^.*?(?:/?\[(?:terminal|OUT-OF-BAND USER MESSAGE)[^\]]*\]).*\n?',
        '', text, flags=re.MULTILINE | re.IGNORECASE,
    )


def _strip_tables_for_tts(text: str) -> str:
    """Remove markdown tables — pipes and separators are unreadable aloud."""
    # Handle multiline tables (each row on its own line)
    text = _TABLE_MULTILINE_RE.sub('', text)
    # Handle single-line tables (all rows concatenated without newlines)
    text = _TABLE_SINGLELINE_RE.sub('', text)
    return text


def strip_source_links_tail(text: str) -> str:
    """Strip trailing source-link appendix blocks commonly found in briefings."""
    return _SOURCE_LINKS_TAIL_RE.sub('', text).rstrip()


def _replace_structured_blocks_for_tts(text: str, config: TtsPipelineConfig) -> str:
    if not config.skip_code_blocks:
        return text
    phrase = config.code_block_phrase.strip() or DEFAULT_CODE_BLOCK_PHRASE
    return _CODE_BLOCK_RE.sub(phrase, text)


def _strip_markdown_syntax(text: str) -> str:
    text = _IMAGE_RE.sub('', text)
    text = _LINK_RE.sub(r'\1', text)
    text = _HEADING_RE.sub('', text)
    text = _HR_RE.sub('', text)
    text = _BOLD_ITALIC_RE.sub(r'\1', text)
    text = _BOLD_RE.sub(r'\1', text)
    text = _ITALIC_STAR_RE.sub(r'\1', text)
    text = _BOLD_UNDERSCORE_RE.sub(r'\1', text)
    text = _ITALIC_UNDERSCORE_RE.sub(r'\1', text)
    text = _BLOCKQUOTE_RE.sub('', text)
    text = _UNORDERED_LIST_RE.sub('', text)
    text = _ORDERED_LIST_RE.sub('', text)
    # Tables are unreadable aloud — strip them entirely (pipes, separators, data rows)
    text = _strip_tables_for_tts(text)
    return text


def _apply_emoji_expansion(text: str, mode: EmojiHandling) -> str:
    if mode != 'describe':
        return text
    return _EMOJI_RE.sub(lambda m: EMOJI_DESCRIPTIONS.get(m.group(0), ''), text)


def _apply_emoji_strip(text: str, mode: EmojiHandling) -> str:
    if mode == 'keep':
        return text
    return _EMOJI_RE.sub('', text)


def _protect_tokens(text: str) -> tuple[str, dict[str, str]]:
    token_map: dict[str, str] = {}
    counter = 0

    def protect(pattern: re.Pattern[str], value_fn=None) -> None:
        nonlocal text, counter

        def replace(match: re.Match[str]) -> str:
            nonlocal counter
            token = f'\x02PROT_{_alpha_token(counter)}\x03'
            counter += 1
            token_map[token] = value_fn(match) if value_fn else match.group(0)
            return token

        text = pattern.sub(replace, text)

    protect(_INLINE_CODE_RE, lambda m: m.group(1))
    protect(_VERSION_RE)
    protect(_UNIX_PATH_RE)
    protect(_FILENAME_RE)
    return text, token_map


def _alpha_token(counter: int) -> str:
    value = counter
    chars: list[str] = []
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord('a') + remainder))
        if value == 0:
            break
        value -= 1
    return ''.join(reversed(chars))


def _restore_tokens(text: str, token_map: dict[str, str]) -> str:
    for token, value in token_map.items():
        text = text.replace(token, value)
    return text


def _normalize_apostrophes(text: str) -> str:
    return text.replace('\u2019', "'")


def _expand_contractions(text: str) -> str:
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(lambda m, r=replacement: _match_case(m.group(0), r), text)
    return text


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_dot_prefixed_tokens(text: str) -> str:
    return re.sub(
        r'(^|[\s(/])\.([a-z0-9][a-z0-9._-]*)',
        lambda m: f"{m.group(1)}dot {re.sub(r'[._-]+', ' ', m.group(2))}",
        text,
        flags=re.IGNORECASE,
    )


def _protect_punctuation(text: str) -> str:
    for pattern, token, _restore in PUNCT_RULES:
        text = pattern.sub(token, text)
    return text


def _restore_punctuation(text: str) -> str:
    for _pattern, token, restore in PUNCT_RULES:
        text = text.replace(token, restore)
    return text


def _collapse_whitespace(text: str) -> str:
    text = _NEWLINE_PADDING_RE.sub('\n', text)
    text = _INLINE_WHITESPACE_RE.sub(' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _segment_normalized_text(text: str, config: TtsPipelineConfig) -> list[str]:
    trimmed = text.strip()
    if not trimmed:
        return []

    if config.segmentation == 'none':
        return [_restore_punctuation(trimmed)]

    if config.segmentation == 'paragraph':
        raw_segments = re.split(r'\n\s*\n+', trimmed)
        return _normalize_segments(raw_segments, config.max_segment_chars)

    path_protected = _protect_path_dots_for_segmentation(trimmed)
    sentence_matches = _SENTENCE_RE.findall(path_protected)
    if not sentence_matches:
        return [_restore_punctuation(trimmed)]
    restored = [_restore_path_dots_after_segmentation(segment) for segment in sentence_matches]
    return _normalize_segments(restored, config.max_segment_chars)


def _normalize_segments(segments: list[str], max_chars: int) -> list[str]:
    normalized: list[str] = []
    for segment in segments:
        cleaned = segment.strip()
        if not cleaned:
            continue
        for part in _split_long_segment(cleaned, max_chars):
            restored = _restore_punctuation(part.strip())
            if restored:
                normalized.append(restored)
    return normalized


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(' ', 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _protect_path_dots_for_segmentation(text: str) -> str:
    text = re.sub(
        r'(^|[\s(/])(\.[a-z0-9][a-z0-9._-]*)',
        lambda m: f"{m.group(1)}{m.group(2).replace('.', PATH_DOT_TOKEN)}",
        text,
        flags=re.IGNORECASE,
    )

    def replace_filename(match: re.Match[str]) -> str:
        token = match.group(0)
        if re.fullmatch(r'\d+\.\d+', token):
            return token
        return token.replace('.', PATH_DOT_TOKEN)

    return _FILENAME_RE.sub(replace_filename, text)


def _restore_path_dots_after_segmentation(text: str) -> str:
    return text.replace(PATH_DOT_TOKEN, '.')


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
