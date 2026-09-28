"""Tests for the number→words preprocessing engine in app/services/preprocess.py.

This is the most behavior-rich code in the repo (years, currency, phone numbers,
scientific notation, model names, etc.) and previously had no direct test coverage —
it was only exercised indirectly through normalize_tts_text(). These tests pin down
the exact spoken-word output so regex changes can't silently regress it.

All expected strings were verified against the installed engine before being written.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.preprocess import (
    TextPreprocessor,
    expand_currency,
    expand_decades,
    expand_fractions,
    expand_ip_addresses,
    expand_model_names,
    expand_ordinals,
    expand_percentages,
    expand_phone_numbers,
    expand_ranges,
    expand_scale_suffixes,
    expand_scientific_notation,
    expand_time,
    expand_units,
    float_to_words,
    normalize_leading_decimals,
    number_to_words,
    replace_numbers,
)

# A default-configured pipeline (every expansion enabled).
PIPELINE = TextPreprocessor()


# ─────────────────────────────────────────────
# Integer / float → words
# ─────────────────────────────────────────────
def test_number_to_words_small():
    assert number_to_words(85) == 'eighty-five'


def test_number_to_words_hundreds():
    assert number_to_words(1200) == 'twelve hundred'


def test_number_to_words_zero():
    assert number_to_words(0) == 'zero'


def test_float_to_words_preserves_trailing_zeros():
    # "1.50" must read as "one point five zero", not collapse to "one point five".
    assert float_to_words('3.14') == 'three point one four'
    assert float_to_words('1.50') == 'one point five zero'


def test_replace_numbers_does_not_treat_hyphen_as_minus():
    # A hyphen inside a model name is not a minus sign.
    assert replace_numbers('gpt-3') == 'gpt-three'


# ─────────────────────────────────────────────
# Individual expansion stages (exact outputs)
# ─────────────────────────────────────────────
def test_expand_ordinals():
    assert expand_ordinals('the 42nd day') == 'the forty-second day'


def test_expand_percentages():
    assert expand_percentages('up 50% today') == 'up fifty percent today'


def test_expand_currency_scale_suffix():
    # "$85K" → "eighty-five thousand dollars".
    assert expand_currency('$85K budget') == 'eighty-five thousand dollars budget'


def test_expand_time():
    assert expand_time('at 3:45 pm') == 'at three forty-five pm'


def test_expand_ranges():
    assert expand_ranges('pages 10-20') == 'pages ten to twenty'


def test_expand_units():
    assert expand_units('5 GB free') == 'five gigabytes free'


def test_normalize_leading_decimals():
    # Bare ".5" is normalized to "0.5" so downstream regexes see a full number.
    assert normalize_leading_decimals('.5 chance') == '0.5 chance'


def test_expand_scientific_notation():
    assert expand_scientific_notation('1e-4 scale') == 'one times ten to the negative four scale'


def test_expand_scale_suffixes():
    assert expand_scale_suffixes('2M users') == 'two million users'


def test_expand_fractions():
    assert expand_fractions('3/4 done') == 'three quarters done'


def test_expand_decades():
    assert expand_decades('the 90s were great') == 'the nineties were great'


def test_expand_ip_addresses_octet_by_octet():
    # Each octet is read digit-by-digit, not as a number.
    assert (
        expand_ip_addresses('host 192.168.1.1 up')
        == 'host one nine two dot one six eight dot one dot one up'
    )


def test_expand_phone_numbers_digit_by_digit():
    # Phone numbers are read digit-by-digit, not as a range or integer.
    assert expand_phone_numbers('call 555-1234 now') == 'call five five five one two three four now'


# ─────────────────────────────────────────────
# Model-name expansion (regression: hyphenated words must survive)
# ─────────────────────────────────────────────
def test_expand_model_names_rewrites_version():
    assert expand_model_names('GPT-3') == 'GPT 3'


def test_expand_model_names_leaves_hyphenated_words_alone():
    # Regression: the number group is optional, so a token with no digit after the
    # hyphen (e.g. "ninety-five") must be left untouched — not mangled to "None".
    assert expand_model_names('ninety-five') == 'ninety-five'


# ─────────────────────────────────────────────
# End-to-end pipeline integration
# ─────────────────────────────────────────────
def test_pipeline_years_read_as_words():
    # 1995 is read as "nineteen ninety-five" (year form), and must NOT be corrupted
    # by the model-name stage into "...Nonefive".
    assert PIPELINE('born in 1995') == 'born in nineteen ninety-five'


def test_pipeline_round_years():
    assert PIPELINE('the year 2000') == 'the year two thousand'


def test_pipeline_model_names_and_numbers():
    out = PIPELINE('I have 1200 students and pi is 3.14')
    assert out == 'I have twelve hundred students and pi is three point one four'


def test_pipeline_gpt_version_expands():
    assert 'GPT three' in PIPELINE('GPT-3 rocks')
    assert 'gpt three point five' in PIPELINE('gpt-3.5 is out')


def test_pipeline_percent_and_currency_substrings():
    # Spacing around currency can be quirky, so assert on the stable substrings.
    out = PIPELINE('a 40% discount and $85K budget')
    assert 'forty percent' in out
    assert 'eighty-five thousand dollars' in out
