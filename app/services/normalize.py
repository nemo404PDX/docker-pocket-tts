"""
TTS-specific normalization and segmentation helpers.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.services.preprocess import TextPreprocessor

# Child of the 'PocketTTS' base logger (configured by setup_logging). Uses stdlib
# logging directly — importing app.logging_config here would cycle (logging_config →
# config → normalize).
logger = logging.getLogger('PocketTTS.normalize')

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
# pocket-tts generates one chunk per segment and caps each at MAX_TOKEN_PER_CHUNK
# (50 tokens). Past that limit the model stops emitting EOS cleanly and starts
# looping tail words ("side by side by side"). Calibrated against the English
# sentencepiece tokenizer: prose runs ~2.8-3.2 chars/token, so 130 chars keeps a
# segment at roughly 45 tokens or fewer even in token-dense text (acronym
# expansions like "H T T P S" are denser still). _split_long_segment additionally
# prefers weak-punctuation breaks so most long clauses split well under this cap.
MAX_SEGMENT_CHARS = 130
# Hard per-segment token cap. pocket-tts generates one chunk per segment and caps each at
# MAX_TOKEN_PER_CHUNK (50 tokens in text_chunking.py); past that the model stops emitting
# EOS cleanly and loops tail words ("side by side by side"). Character counts are only a
# proxy — hash-dense or acronym-heavy text can exceed 50 tokens well under 130 chars — so
# _normalize_segments applies this cap with the real sentencepiece tokenizer as a final
# pass. Set to 0 to disable (falls back to the character cap alone).
MAX_SEGMENT_TOKENS = int(os.environ.get('POCKET_TTS_MAX_SEGMENT_TOKENS', '50')) or None
PATH_DOT_TOKEN = '\x02PD\x03'
# Weak punctuation used to break long clauses at natural pause points, mirroring
# pocket-tts's own fallback split set (`,;:`) in text_chunking.split_into_best_sentences.
_WEAK_PUNCT = ',;:'

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
# Default marker pattern stripped from TTS text (opening/closing tags + tool output).
# Overridable at runtime via POCKET_TTS_HERMES_MARKER_PATTERN — see _strip_hermes_marker_lines.
DEFAULT_HERMES_MARKER_PATTERN = r'/?\[[/\s]*(?:terminal|OUT-OF-BAND USER MESSAGE)[^\]]*\]'
_FILENAME_RE = re.compile(
    r'(?<!\.)\b[a-z0-9_-]+\.[a-z0-9_-]{2,}(?:\.[a-z0-9._-]+)*\b', re.IGNORECASE
)
# Bare git-style short hashes (5-12 lowercase alnum chars containing at least one digit
# AND one letter), e.g. "68187ab" or each half of a range like "52c36f7..68187ab".
# Clients such as Open WebUI strip markdown before sending text to TTS, so backticked
# hashes arrive bare and the inline-code protection never fires — number expansion then
# mangles them ("sixty-eight thousand one hundred eighty-sevenab"), which makes the model
# skip or garble the surrounding words. Requiring both a digit and a letter (in either
# order) keeps normal words ("master", "remotes") and pure numbers ("12345" → "twelve
# thousand...") untouched, while catching hashes that start with a digit like "68187ab".
_BARE_HASH_RE = re.compile(r'\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{5,12}\b', re.IGNORECASE)
# Mixed alphanumeric tokens containing digits (usernames, repo names: "nemo404PDX").
# Without this protection the digit runs get expanded ("nemo four hundred four P D X"),
# which both garbles the reading and inflates token counts past pocket-tts's 50-token cap.
_MIXED_ALNUM_RE = re.compile(r'\b[A-Za-z]+\d[A-Za-z0-9]*\b')
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

# All-caps acronyms/initialisms that TTS models mangle when read as a word ("IP" ->
# "eye-py", "CPU" -> "soo-oo"). Expanded to their spoken form so every voice says them
# consistently. Matching is case-sensitive on the all-caps token, so mixed-case words like
# "ffmpeg" or "iPhone" are left alone (add them here explicitly if you want a specific reading).
ACRONYM_EXPANSIONS: dict[str, str] = {
    # Spoken as individual letters (spaced so the model reads each letter).
    'AI': 'A I', 'API': 'A P I', 'CPU': 'C P U', 'GPU': 'G P U', 'IP': 'I P',
    'URL': 'U R L', 'URI': 'U R I', 'UUID': 'U U I D', 'JSON': 'J S O N',
    'XML': 'X M L', 'HTML': 'H T M L', 'CSS': 'C S S', 'SQL': 'S Q L',
    'SSH': 'S S H', 'DNS': 'D N S', 'HTTP': 'H T T P', 'HTTPS': 'H T T P S',
    'TCP': 'T C P', 'UDP': 'U D P', 'LAN': 'L A N', 'WAN': 'W A N',
    'RAM': 'R A M', 'ROM': 'R O M', 'USB': 'U S B', 'HDD': 'H D D',
    'SSD': 'S S D', 'LED': 'L E D', 'LCD': 'L C D', 'TV': 'T V',
    'PC': 'P C', 'OS': 'O S', 'CLI': 'C L I', 'SDK': 'S D K',
    'IDE': 'I D E', 'CI': 'C I', 'CDN': 'C D N', 'OTA': 'O T A',
    'FAQ': 'F A Q', 'DIY': 'D I Y', 'FYI': 'F Y I', 'IMO': 'I M O',
    'ETA': 'E T A', 'CEO': 'C E O', 'CTO': 'C T O', 'COO': 'C O O',
    'KPI': 'K P I', 'ROI': 'R O I', 'SLA': 'S L A', 'SLO': 'S L O',
    'TTS': 'text to speech', 'STT': 'speech to text', 'LLM': 'large language model',
    'NLP': 'natural language processing', 'ML': 'machine learning',
}

# Precompiled: word-boundary, case-sensitive (all-caps tokens only). Longer terms first so
# e.g. "HTTPS" isn't partially eaten by "HTTP".
_ACRONYM_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf'\b{term}\b'), spoken)
    for term, spoken in sorted(ACRONYM_EXPANSIONS.items(), key=lambda kv: -len(kv[0]))
    if spoken
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
    # Hard per-segment token cap (pocket-tts's 50-token chunk limit). None disables the
    # tokenizer-based pass and relies on max_segment_chars alone.
    max_segment_tokens: int | None = MAX_SEGMENT_TOKENS
    run_number_expansion: bool = True
    drop_source_links_section: bool = False
    # Regex of client markers to strip (whole line removed). None → default pattern.
    marker_pattern: str | None = None


@dataclass(slots=True)
class TtsNormDebugResult:
    original: str
    normalized: str
    segments: list[str]
    steps: list[tuple[StageName, str]] = field(default_factory=list)


# Only the number-expansion stages are enabled; everything else in this project's
# pipeline is handled by the other normalisation steps (see below). All of these
# are TextPreprocessor defaults — listed explicitly to document intent.
_NUMBER_PREPROCESSOR = TextPreprocessor(
    replace_numbers=True,
    replace_floats=True,
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
    remove_html=True,
    normalize_unicode=True,
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
    # Strip client marker lines after tables so [terminal] inside table cells is already gone
    result = _strip_hermes_marker_lines(result, config.marker_pattern)
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
    protected_text = _expand_acronyms(protected_text)
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


def _strip_hermes_marker_lines(text: str, marker_pattern: str | None = None) -> str:
    """Remove entire lines that contain client markers like [terminal] or [OUT-OF-BAND].

    ``marker_pattern`` semantics:
      - ``None``            → use DEFAULT_HERMES_MARKER_PATTERN (the Hermes default).
      - explicitly empty '' → disable marker stripping entirely.
      - a valid regex       → used as-is (case-insensitive); the whole line is removed.
      - an invalid regex    → log a warning and leave the text unchanged.

    This keeps the server decoupled from any one client's output format — override via
    POCKET_TTS_HERMES_MARKER_PATTERN if you feed a different client's transcript.
    """
    if marker_pattern is None:
        pattern = DEFAULT_HERMES_MARKER_PATTERN
    else:
        pattern = marker_pattern.strip()
        if not pattern:  # explicitly empty → disabled
            return text

    try:
        re.compile(pattern, re.IGNORECASE)  # validate; the raw string is embedded below
    except re.error as exc:
        logger.warning('Invalid POCKET_TTS_HERMES_MARKER_PATTERN %r (%s); skipping marker stripping', pattern, exc)
        return text

    # Match a full line (start to newline) containing the marker. The trailing .* is greedy
    # so it captures everything after the marker on the same line. Embed the raw pattern —
    # interpolating a compiled object would stringify to "re.compile(...)", not the regex.
    return re.sub(rf'^.*?(?:{pattern}).*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)


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
            # No underscore in the token: _ITALIC_UNDERSCORE_RE / _BOLD_UNDERSCORE_RE match
            # on '_' pairs and would eat the separator of any two protected tokens in the
            # same text, making them unrecoverable (literal \x02PROT..\x03 garbage then
            # reaches the TTS model, which skips or garbles the surrounding words).
            token = f'\x02PROT{_alpha_token(counter)}\x03'
            counter += 1
            token_map[token] = value_fn(match) if value_fn else match.group(0)
            return token

        text = pattern.sub(replace, text)

    protect(_INLINE_CODE_RE, lambda m: m.group(1))
    protect(_VERSION_RE)
    protect(_UNIX_PATH_RE)
    protect(_FILENAME_RE)
    protect(_BARE_HASH_RE)
    protect(_MIXED_ALNUM_RE)
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


def _expand_acronyms(text: str) -> str:
    """Expand all-caps acronyms to their spoken form so TTS reads them consistently.

    Case-sensitive on the all-caps token, so mixed-case words (ffmpeg, iPhone) are left
    alone. Longer terms are applied first so "HTTPS" isn't partially eaten by "HTTP".
    """
    for pattern, replacement in _ACRONYM_RULES:
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
        return _normalize_segments(raw_segments, config.max_segment_chars, config.max_segment_tokens)

    path_protected = _protect_path_dots_for_segmentation(trimmed)
    sentence_matches = _SENTENCE_RE.findall(path_protected)
    if not sentence_matches:
        return [_restore_punctuation(trimmed)]
    restored = [_restore_path_dots_after_segmentation(segment) for segment in sentence_matches]
    return _normalize_segments(restored, config.max_segment_chars, config.max_segment_tokens)


def _normalize_segments(segments: list[str], max_chars: int, max_tokens: int | None = None) -> list[str]:
    normalized: list[str] = []
    for segment in segments:
        cleaned = segment.strip()
        if not cleaned:
            continue
        for part in _split_long_segment(cleaned, max_chars):
            restored = _restore_punctuation(part.strip())
            if not restored:
                continue
            # Character caps are only a proxy for token counts — hash-dense or
            # acronym-heavy text can exceed pocket-tts's 50-token chunk cap well under
            # the character limit. Enforce the real cap with the model's own tokenizer.
            if max_tokens:
                normalized.extend(_split_segment_by_tokens(restored, max_tokens))
            else:
                normalized.append(restored)
    return normalized


_TOKENIZER_CACHE: dict[str, object] = {}


def _load_tokenizer() -> object | None:
    """Load pocket-tts's sentencepiece tokenizer (cached). Returns None if unavailable.

    Searches the configured model cache dir first, then the repo-local data dir and the
    HuggingFace cache — wherever `hf download kyutai/pocket-tts` may have landed it.
    """
    if 'sp' in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE['sp']

    candidates = [Path(p) for p in (os.environ.get('POCKET_TTS_MODEL_CACHE_DIR'), './data/models/pocket-tts') if p]
    candidates.append(Path.home() / '.cache' / 'huggingface')
    model_path: Path | None = None
    for root in candidates:
        try:
            matches = sorted(root.rglob('tokenizer.model'))
        except OSError:
            continue
        if matches:
            model_path = matches[0]
            break

    tokenizer = None
    if model_path is not None:
        try:
            from sentencepiece import SentencePieceProcessor

            sp = SentencePieceProcessor()
            sp.load(str(model_path))
            tokenizer = sp
        except Exception as exc:  # pragma: no cover - depends on installed deps
            logger.warning('Could not load sentencepiece tokenizer %s (%s); token cap disabled', model_path, exc)

    _TOKENIZER_CACHE['sp'] = tokenizer
    return tokenizer


def _count_tokens(text: str) -> int | None:
    """Token count under pocket-tts's own tokenizer, or None if unavailable."""
    sp = _load_tokenizer()
    if sp is None or not text.strip():
        return None
    try:
        return len(sp.encode(text))
    except Exception:  # pragma: no cover - defensive; tokenizer errors shouldn't kill TTS
        return None


def _split_segment_by_tokens(segment: str, max_tokens: int) -> list[str]:
    """Split a segment so every part is at most ``max_tokens`` tokens.

    Greedy word packing: keep appending words while the running total stays under the cap.
    A single unbreakable token longer than the cap is returned as-is (nothing to do).
    """
    if _count_tokens(segment) in (None, 0):
        return [segment] if segment.strip() else []

    parts: list[str] = []
    current: list[str] = []
    for word in segment.split():
        trial_count = _count_tokens(' '.join(current + [word]))
        if current and trial_count is not None and trial_count > max_tokens:
            parts.append(' '.join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(' '.join(current))
    return parts


def _split_long_segment(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = _find_split_point(remaining, max_chars)
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _find_split_point(text: str, max_chars: int) -> int:
    """Pick where to cut a segment that exceeds ``max_chars``.

    Preference order (all within the first ``max_chars`` characters):
      1. Just after weak punctuation (`,;:`) — natural pause points, mirroring
         pocket-tts's own fallback split set in text_chunking.split_into_best_sentences.
         Only considered in the latter half of the window so we don't produce tiny
         fragments that would make the audio choppy.
      2. The last space before ``max_chars`` (word boundary).
      3. A hard cut at ``max_chars`` (e.g., one unbroken token longer than the cap).

    Ending a segment on weak punctuation is safe downstream: pocket-tts's
    _ensure_terminal_punctuation replaces a trailing comma/colon with a period, so
    each chunk still gets clean sentence-final intonation.
    """
    min_split = max(1, max_chars // 2)
    for idx in range(max_chars - 1, min_split - 1, -1):
        if text[idx] in _WEAK_PUNCT:
            return idx + 1
    split_at = text.rfind(' ', 0, max_chars)
    if split_at <= 0:
        split_at = max_chars
    return split_at


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
