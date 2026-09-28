"""
Adapted and supplemented from origional at https://github.com/KittenML/KittenTTS/blob/main/kittentts/preprocess.py
See license at: https://github.com/KittenML/KittenTTS/blob/main/LICENSE (Apache 2.0)

Trimmed to the stages this project actually uses (see _NUMBER_PREPROCESSOR in
normalize.py). Unused KittenTTS helpers (URL/email/hashtag removal, lowercase,
stopwords, symbol expansion, roman numerals, month expansion, ...) were removed;
restore from git history or upstream if ever needed.
"""

import re
import unicodedata

# ─────────────────────────────────────────────
# Number → Words conversion
# ─────────────────────────────────────────────

_ONES = [
    '',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
]
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_SCALE = ['', 'thousand', 'million', 'billion', 'trillion']

_ORDINAL_EXCEPTIONS = {
    'one': 'first',
    'two': 'second',
    'three': 'third',
    'four': 'fourth',
    'five': 'fifth',
    'six': 'sixth',
    'seven': 'seventh',
    'eight': 'eighth',
    'nine': 'ninth',
    'twelve': 'twelfth',
}

_CURRENCY_SYMBOLS = {
    '$': 'dollar',
    '€': 'euro',
    '£': 'pound',
    '¥': 'yen',
    '₹': 'rupee',
    '₩': 'won',
    '₿': 'bitcoin',
}

_CURRENCY_SCALE_MAP = {
    'K': 'thousand',
    'M': 'million',
    'B': 'billion',
    'T': 'trillion',
    'thousand': 'thousand',
    'million': 'million',
    'billion': 'billion',
    'trillion': 'trillion',
}


def _three_digits_to_words(n: int) -> str:
    """Convert a number 0-999 to English words."""
    if n == 0:
        return ''
    parts = []
    hundreds = n // 100
    remainder = n % 100
    if hundreds:
        parts.append(f'{_ONES[hundreds]} hundred')
    if remainder < 20:
        if remainder:
            parts.append(_ONES[remainder])
    else:
        tens_word = _TENS[remainder // 10]
        ones_word = _ONES[remainder % 10]
        parts.append(f'{tens_word}-{ones_word}' if ones_word else tens_word)
    return ' '.join(parts)


def number_to_words(n: int) -> str:
    """
    Convert an integer to its English word representation.

    Examples:
        1200      → "twelve hundred"
        1000      → "one thousand"
        1_000_000 → "one million"
        -42       → "negative forty-two"
        0         → "zero"
    """
    if not isinstance(n, int):
        n = int(n)
    if n == 0:
        return 'zero'
    if n < 0:
        return f'negative {number_to_words(-n)}'

    # X00-X999 read as "X hundred" (e.g. 1200 → "twelve hundred")
    # Exclude exact multiples of 1000 (1000 → "one thousand", not "ten hundred")
    if 100 <= n <= 9999 and n % 100 == 0 and n % 1000 != 0:
        hundreds = n // 100
        if hundreds < 20:
            return f'{_ONES[hundreds]} hundred'

    parts = []
    for _i, scale in enumerate(_SCALE):
        chunk = n % 1000
        if chunk:
            chunk_words = _three_digits_to_words(chunk)
            parts.append(f'{chunk_words} {scale}'.strip() if scale else chunk_words)
        n //= 1000
        if n == 0:
            break

    return ' '.join(reversed(parts))


def float_to_words(value, decimal_sep: str = 'point') -> str:
    """
    Convert a float (or numeric string) to words, reading decimal digits individually.
    Accepts a string to preserve trailing zeros (e.g. "1.50" → "one point five zero").

    Examples:
        3.14   → "three point one four"
        -0.5   → "negative zero point five"
        "3.10" → "three point one zero"
        1.007  → "one point zero zero seven"
    """
    text = value if isinstance(value, str) else f'{value}'
    negative = text.startswith('-')
    if negative:
        text = text[1:]

    if '.' in text:
        int_part, dec_part = text.split('.', 1)
        int_words = number_to_words(int(int_part)) if int_part else 'zero'
        # Read each decimal digit individually; "0" → "zero"
        digit_map = ['zero'] + _ONES[1:]  # index 0 → "zero"
        dec_words = ' '.join(digit_map[int(d)] for d in dec_part)
        result = f'{int_words} {decimal_sep} {dec_words}'
    else:
        result = number_to_words(int(text))

    return f'negative {result}' if negative else result


# ─────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────

_RE_HTML = re.compile(r'<[^>]+>')

# Number: do NOT match a leading minus if it is immediately preceded by a letter
# (handles "gpt-3", "gpl-3", "v-2" etc.)
_RE_NUMBER = re.compile(r'(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?')

# Ordinals: 1st, 2nd, 3rd, 4th … 21st, 101st …
_RE_ORDINAL = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

# Percentages: 50%, 3.5%
_RE_PERCENT = re.compile(r'(-?[\d,]+(?:\.\d+)?)\s*%')

# Currency: $100, €1,200.50, £50, $85K, $2.5M (optional scale suffix)
_RE_CURRENCY = re.compile(
    r'([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*(million|billion|trillion|thousand|[KMBT])?\b',
    re.IGNORECASE,
)

# Time: 3:30pm, 14:00, 3:30 AM — requires 2-digit minutes so "3:0" (score) doesn't match
_RE_TIME = re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b', re.IGNORECASE)

# Ranges: 10-20, 100-200 (both sides numeric, hyphen between them)
_RE_RANGE = re.compile(r'(?<!\w)(\d+)-(\d+)(?!\w)')

# Version/model names: gpt-3, gpt-3.5, v2.0, Python-3.10, GPL-3
# Letter(s) + hyphen + digit(s) [+ more version parts]
_RE_MODEL_VER = re.compile(r'\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)?(?=[^\d.]|$)')

# Measurement units glued to numbers: 100km, 50kg, 25°C, 5GB
_RE_UNIT = re.compile(
    r'(\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b',
    re.IGNORECASE,
)

# Scale suffixes (uppercase only to avoid ambiguity): 7B, 340M, 1.5K, 2T
# Must NOT be preceded by a letter (so 'MB' is handled by unit regex first)
_RE_SCALE = re.compile(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])')

# Scientific notation: 1e-4, 2.5e10, 6.022E23
_RE_SCI = re.compile(r'(?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![a-zA-Z\d])')

# Fractions: 1/2, 3/4, 2/3
_RE_FRACTION = re.compile(r'\b(\d+)\s*/\s*(\d+)\b')

# Decades: 80s, 90s, 1980s, 2020s (number ending in 0 followed by 's')
_RE_DECADE = re.compile(r'\b(\d{1,3})0s\b')

# Leading decimal (no digit before the dot): .5, .75
_RE_LEAD_DEC = re.compile(r'(?<!\d)\.([\d])')

# Years: 4-digit numbers in range 1000-2999 (standalone, not part of larger number)
_RE_YEAR = re.compile(r'\b([1][0-9]{3})\b')


# ─────────────────────────────────────────────
# Expansion helpers
# ─────────────────────────────────────────────
def _year_to_words(year: int) -> str:
    """Convert a year to its spoken form.

    Examples:
        1977 → "nineteen seventy-seven"
        2005 → "two thousand five"
        1000 → "ten hundred"
        2024 → "twenty twenty-four"
    """
    if year < 1000 or year > 9999:
        return number_to_words(year)

    # Years in the format XXYY (e.g. 1977, 2024)
    first = year // 100
    last = year % 100

    if last == 0 and year >= 1000:
        # e.g. 2000 → "two thousand", 1900 → "nineteen hundred"
        return number_to_words(year)
    elif first < 20:
        # e.g. 1977 → "nineteen seventy-seven"
        first_word = _ONES[first] if first < len(_ONES) else number_to_words(first)
        last_word = 'oh' + (' ' + _ONES[last]) if last < 10 and last > 0 else (number_to_words(last) if last else '')
        return f'{first_word} {last_word}'.strip()
    else:
        # e.g. 2024 → "twenty twenty-four"
        first_word = number_to_words(first)
        last_word = 'oh' + (' ' + _ONES[last]) if last < 10 and last > 0 else (number_to_words(last) if last else '')
        return f'{first_word} {last_word}'.strip()


def expand_years(text: str) -> str:
    """Convert standalone years to their spoken form.

    Examples:
        "in 1977" → "in nineteen seventy-seven"
        "year 2024" → "year twenty twenty-four"
    """
    def _replace(m: re.Match) -> str:
        year = int(m.group(1))
        return _year_to_words(year)

    return _RE_YEAR.sub(_replace, text)


def _ordinal_suffix(n: int) -> str:
    """Return the ordinal word for n (e.g. 1 → 'first', 5 → 'fifth', 21 → 'twenty-first')."""
    word = number_to_words(n)
    # For hyphenated compounds like "twenty-one", convert only the last part
    if '-' in word:
        prefix, last = word.rsplit('-', 1)
        joiner = '-'
    else:
        parts = word.rsplit(' ', 1)
        prefix, last, joiner = (parts[0], parts[1], ' ') if len(parts) == 2 else ('', parts[0], '')

    # Check exception table
    for base, ordinal in _ORDINAL_EXCEPTIONS.items():
        if last == base:
            last_ord = ordinal
            break
    else:
        # General rule
        if last.endswith('t'):
            last_ord = last + 'h'
        elif last.endswith('e'):
            last_ord = last[:-1] + 'th'
        else:
            last_ord = last + 'th'

    return f'{prefix}{joiner}{last_ord}' if prefix else last_ord


def expand_ordinals(text: str) -> str:
    """
    Convert ordinal numbers to words.

    Examples:
        "1st place"  → "first place"
        "2nd floor"  → "second floor"
        "3rd base"   → "third base"
        "21st century" → "twenty-first century"
        "100th day"  → "one hundredth day"
    """

    def _replace(m: re.Match) -> str:
        return _ordinal_suffix(int(m.group(1)))

    return _RE_ORDINAL.sub(_replace, text)


def expand_percentages(text: str) -> str:
    """
    Expand percentage expressions.

    Examples:
        "50% off"    → "fifty percent off"
        "3.5% rate"  → "three point five percent rate"
        "-2% change" → "negative two percent change"
    """

    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(',', '')
        if '.' in raw:
            return float_to_words(float(raw)) + ' percent'
        return number_to_words(int(raw)) + ' percent'

    return _RE_PERCENT.sub(_replace, text)


def expand_currency(text: str) -> str:
    """
    Expand currency amounts, including optional scale suffixes.

    Examples:
        "$100"      → "one hundred dollars"
        "€1,200.50" → "twelve hundred euros and fifty cents"
        "£9.99"     → "nine pounds and ninety-nine cents"
        "$85K"      → "eighty five thousand dollars"
        "$2.5M"     → "two point five million dollars"
    """

    def _replace(m: re.Match) -> str:
        symbol = m.group(1)
        raw = m.group(2).replace(',', '')
        scale_suffix = m.group(3)
        unit = _CURRENCY_SYMBOLS.get(symbol, '')

        # Handle Scaled Currency ($17.5 billion or $17.5B)
        if scale_suffix:
            # Normalize suffix (e.g., 'B' or 'billion' -> 'billion')
            scale_word = _CURRENCY_SCALE_MAP.get(scale_suffix.upper(), scale_suffix.lower())
            num = float_to_words(raw) if '.' in raw else number_to_words(int(raw))
            return f'{num} {scale_word} {unit}{"s" if unit else ""}'.strip()

        # Handle Standard Currency ($17.50)
        if '.' in raw:
            int_part, dec_part = raw.split('.', 1)
            dec_val = int(dec_part[:2].ljust(2, '0'))
            int_words = number_to_words(int(int_part))
            result = f'{int_words} {unit}s' if unit else int_words
            if dec_val:
                cents = number_to_words(dec_val)
                result += f' and {cents} cent{"s" if dec_val != 1 else ""}'
        else:
            val = int(raw)
            words = number_to_words(val)
            result = f'{words} {unit}{"s" if val != 1 and unit else ""}' if unit else words
        return result

    return _RE_CURRENCY.sub(_replace, text)


def expand_time(text: str) -> str:
    """
    Expand time expressions.

    Examples:
        "3:30pm"  → "three thirty pm"
        "14:00"   → "fourteen hundred"
        "9:05 AM" → "nine oh five am"
        "12:00pm" → "twelve pm"
    """

    def _replace(m: re.Match) -> str:
        h = int(m.group(1))
        mins = int(m.group(2))
        suffix = (' ' + m.group(4).lower()) if m.group(4) else ''
        h_words = number_to_words(h)
        if mins == 0:
            return f'{h_words} hundred{suffix}' if not m.group(4) else f'{h_words}{suffix}'
        elif mins < 10:
            return f'{h_words} oh {number_to_words(mins)}{suffix}'
        else:
            return f'{h_words} {number_to_words(mins)}{suffix}'

    return _RE_TIME.sub(_replace, text)


def expand_ranges(text: str) -> str:
    """
    Expand numeric ranges.

    Examples:
        "10-20 items"   → "ten to twenty items"
        "pages 100-200" → "pages one hundred to two hundred"
        "2020-2024"     → "twenty twenty to twenty twenty-four"
    """

    def _replace(m: re.Match) -> str:
        lo = number_to_words(int(m.group(1)))
        hi = number_to_words(int(m.group(2)))
        return f'{lo} to {hi}'

    return _RE_RANGE.sub(_replace, text)


def expand_model_names(text: str) -> str:
    """
    Normalise version/model names that use letter-hyphen-number patterns,
    so the number is not misread as negative.

    Examples:
        "GPT-3"      → "GPT 3"
        "gpt-3.5"    → "gpt 3.5"
        "GPL-3"      → "GPL 3"
        "Python-3.10"→ "Python 3.10"
        "v2.0"       stays as "v2.0" (no hyphen — handled by number replacement)
        "IPv6"       stays as "IPv6"
    """
    return _RE_MODEL_VER.sub(lambda m: f'{m.group(1)} {m.group(2)}', text)


def expand_units(text: str) -> str:
    """
    Expand common measurement units glued to numbers.

    Examples:
        "100km"  → "one hundred kilometers"
        "50kg"   → "fifty kilograms"
        "25°C"   → "twenty-five degrees Celsius"
        "5GB"    → "five gigabytes"
    """
    _unit_map = {
        'km': 'kilometers',
        'kg': 'kilograms',
        'mg': 'milligrams',
        'ml': 'milliliters',
        'gb': 'gigabytes',
        'mb': 'megabytes',
        'kb': 'kilobytes',
        'tb': 'terabytes',
        'hz': 'hertz',
        'khz': 'kilohertz',
        'mhz': 'megahertz',
        'ghz': 'gigahertz',
        'mph': 'miles per hour',
        'kph': 'kilometers per hour',
        'ms': 'milliseconds',
        'ns': 'nanoseconds',
        'µs': 'microseconds',
        '°c': 'degrees Celsius',
        'c°': 'degrees Celsius',
        '°f': 'degrees Fahrenheit',
        'f°': 'degrees Fahrenheit',
    }

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        unit = m.group(2).lower()
        expanded = _unit_map.get(unit, m.group(2))
        num = float_to_words(float(raw)) if '.' in raw else number_to_words(int(raw))
        return f'{num} {expanded}'

    return _RE_UNIT.sub(_replace, text)


def normalize_leading_decimals(text: str) -> str:
    """
    Normalise bare leading-decimal floats so the number pipeline handles them.

    Examples:
        ".5 teaspoons" → "0.5 teaspoons"
        "-.25 adjustment" → "-0.25 adjustment"
    """
    # Handle -.5 → -0.5 and .5 → 0.5
    text = re.sub(r'(?<!\d)(-)\.([\d])', r'\g<1>0.\2', text)
    return _RE_LEAD_DEC.sub(r'0.\1', text)


def expand_scientific_notation(text: str) -> str:
    """
    Expand scientific-notation numbers to spoken form.

    Examples:
        "1e-4"    → "one times ten to the negative four"
        "2.5e10"  → "two point five times ten to the ten"
        "6.022E23"→ "six point zero two two times ten to the twenty three"
    """

    def _replace(m: re.Match) -> str:
        coeff_raw = m.group(1)
        exp = int(m.group(2))
        coeff_words = (
            float_to_words(coeff_raw) if '.' in coeff_raw else number_to_words(int(coeff_raw))
        )
        exp_words = number_to_words(abs(exp))
        sign = 'negative ' if exp < 0 else ''
        return f'{coeff_words} times ten to the {sign}{exp_words}'

    return _RE_SCI.sub(_replace, text)


def expand_scale_suffixes(text: str) -> str:
    """
    Expand standalone uppercase scale suffixes attached to numbers.

    Examples:
        "7B parameters" → "seven billion parameters"
        "340M model"    → "three hundred forty million model"
        "1.5K salary"   → "one point five thousand salary"
        "$100K budget"  → "$100K budget"  (currency handled upstream)
    """
    _map = {'K': 'thousand', 'M': 'million', 'B': 'billion', 'T': 'trillion'}

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        suffix = m.group(2)
        scale_word = _map.get(suffix, suffix)
        num = float_to_words(raw) if '.' in raw else number_to_words(int(raw))
        return f'{num} {scale_word}'

    return _RE_SCALE.sub(_replace, text)


def expand_fractions(text: str) -> str:
    """
    Expand simple numeric fractions.

    Examples:
        "1/2 cup"  → "one half cup"
        "3/4 mile" → "three quarters mile"
        "2/3 done" → "two thirds done"
        "5/8 inch" → "five eighths inch"
    """

    def _replace(m: re.Match) -> str:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return m.group()
        num_words = number_to_words(num)
        if den == 2:
            denom_word = 'half' if num == 1 else 'halves'
        elif den == 4:
            denom_word = 'quarter' if num == 1 else 'quarters'
        else:
            denom_word = _ordinal_suffix(den)
            if num != 1:
                denom_word += 's'
        return f'{num_words} {denom_word}'

    return _RE_FRACTION.sub(_replace, text)


def expand_decades(text: str) -> str:
    """
    Expand decade expressions to words.

    Examples:
        "the 80s"    → "the eighties"
        "the 1980s"  → "the nineteen eighties"
        "the 2020s"  → "the twenty twenties"
        "'90s music" → "nineties music"
    """
    _decade_map = {
        0: 'hundreds',
        1: 'tens',
        2: 'twenties',
        3: 'thirties',
        4: 'forties',
        5: 'fifties',
        6: 'sixties',
        7: 'seventies',
        8: 'eighties',
        9: 'nineties',
    }

    def _replace(m: re.Match) -> str:
        base = int(m.group(1))  # e.g. 8 for "80s", 198 for "1980s"
        decade_digit = base % 10
        decade_word = _decade_map.get(decade_digit, '')
        if base < 10:
            return decade_word
        century_part = base // 10  # e.g. 19 for 198
        return f'{number_to_words(century_part)} {decade_word}'

    return _RE_DECADE.sub(_replace, text)


def expand_ip_addresses(text: str) -> str:
    """
    Expand IPv4 addresses to spoken digits per octet.

    Examples:
        "192.168.1.1"  → "one nine two dot one six eight dot one dot one"
        "10.0.0.1"     → "one zero dot zero dot zero dot one"
    """
    _d = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }

    def _octet(s: str) -> str:
        return ' '.join(_d[c] for c in s)

    def _replace(m: re.Match) -> str:
        return ' dot '.join(_octet(g) for g in m.groups())

    return re.sub(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', _replace, text)


def expand_phone_numbers(text: str) -> str:
    """
    Expand US phone numbers to spoken digits before range expansion claims the hyphens.

    Examples:
        "555-1234"       → "five five five one two three four"
        "555-123-4567"   → "five five five one two three four five six seven"
        "1-800-555-0199" → "one eight zero zero five five five zero one nine nine"
    """
    _d = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }

    def _digits(s: str) -> str:
        return ' '.join(_d[c] for c in s)

    def _join(*groups) -> str:
        return ' '.join(_digits(g) for g in groups)

    # Match longest pattern first to avoid partial matches
    # 11-digit: 1-800-555-0199
    text = re.sub(
        r'(?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!\-\d)',
        lambda m: _join(*m.groups()),
        text,
    )
    # 10-digit: 555-123-4567
    text = re.sub(
        r'(?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!\-\d)', lambda m: _join(*m.groups()), text
    )
    # 7-digit local: 555-1234 (not preceded or followed by digit-hyphen to avoid sub-matching)
    text = re.sub(r'(?<!\d-)\b(\d{3})-(\d{4})\b(?!\-\d)', lambda m: _join(*m.groups()), text)
    return text


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text."""
    return _RE_HTML.sub(' ', text)


def normalize_unicode(text: str, form: str = 'NFC') -> str:
    """Normalize unicode characters (NFC, NFD, NFKC, or NFKD)."""
    return unicodedata.normalize(form, text)


# ─────────────────────────────────────────────
# Core preprocessing functions
# ─────────────────────────────────────────────

def replace_numbers(text: str, replace_floats: bool = True) -> str:
    """
    Replace all numeric tokens with their word equivalents.

    Examples:
        "There are 1200 students" → "There are twelve hundred students"
        "Pi is 3.14"              → "Pi is three point one four"
        "gpt-3 rocks"             → "gpt-3 rocks"  (hyphen not treated as minus)
    """

    def _replace(m: re.Match) -> str:
        raw = m.group().replace(',', '')
        try:
            if '.' in raw and replace_floats:
                # Pass raw string so trailing zeros are preserved ("1.50" → "one point five zero")
                return float_to_words(raw)
            else:
                return number_to_words(int(float(raw)))
        except (ValueError, OverflowError):
            return m.group()

    return _RE_NUMBER.sub(_replace, text)


# ─────────────────────────────────────────────
# Pipeline helper
# ─────────────────────────────────────────────


class TextPreprocessor:
    """
    Configurable preprocessing pipeline for TTS number expansion.

    Usage:
        pp = TextPreprocessor()
        clean = pp("GPT-3 costs $0.002 per token — 50% cheaper than before!")
        # → "GPT three costs zero dollars and zero point two cents per token fifty percent cheaper than before!"
    """

    def __init__(
        self,
        replace_numbers: bool = True,
        replace_floats: bool = True,
        expand_model_names: bool = True,
        expand_ordinals: bool = True,
        expand_percentages: bool = True,
        expand_currency: bool = True,
        expand_time: bool = True,
        expand_ranges: bool = True,
        expand_units: bool = True,
        expand_scale_suffixes: bool = True,
        expand_scientific_notation: bool = True,
        expand_fractions: bool = True,
        expand_decades: bool = True,
        expand_phone_numbers: bool = True,
        expand_ip_addresses: bool = True,
        expand_years: bool = True,
        normalize_leading_decimals: bool = True,
        remove_html: bool = True,
        normalize_unicode: bool = True,
    ):
        self.config = {k: v for k, v in locals().items() if k != 'self'}

    def __call__(self, text: str) -> str:
        return self.process(text)

    def process(self, text: str) -> str:
        cfg = self.config
        if cfg['normalize_unicode']:
            text = normalize_unicode(text)
        if cfg['remove_html']:
            text = remove_html_tags(text)
        # IP addresses before normalize_leading_decimals (IPs contain dots before digits)
        # Years before generic number replacement
        if cfg['expand_years']:
            text = expand_years(text)
        # IP addresses before normalize_leading_decimals (IPs contain dots before digits)
        if cfg['expand_ip_addresses']:
            text = expand_ip_addresses(text)
        # Normalise bare leading decimals early so downstream regexes see "0.5" not ".5"
        if cfg['normalize_leading_decimals']:
            text = normalize_leading_decimals(text)
        # Expand special forms before generic number replacement
        if cfg['expand_currency']:
            text = expand_currency(text)
        if cfg['expand_percentages']:
            text = expand_percentages(text)
        # Scientific notation before model-name expansion (e.g. "1e-4" contains "e-4")
        if cfg['expand_scientific_notation']:
            text = expand_scientific_notation(text)
        if cfg['expand_time']:
            text = expand_time(text)
        if cfg['expand_ordinals']:
            text = expand_ordinals(text)
        if cfg['expand_units']:
            text = expand_units(text)
        # Scale suffixes after units (units handles "MB"/"GB"; this handles bare "B"/"M")
        if cfg['expand_scale_suffixes']:
            text = expand_scale_suffixes(text)
        if cfg['expand_fractions']:
            text = expand_fractions(text)
        if cfg['expand_decades']:
            text = expand_decades(text)
        # Phone numbers before ranges, otherwise NNN-NNNN is treated as a range
        if cfg['expand_phone_numbers']:
            text = expand_phone_numbers(text)
        if cfg['expand_ranges']:
            text = expand_ranges(text)
        if cfg['expand_model_names']:
            text = expand_model_names(text)
        if cfg['replace_numbers']:
            text = replace_numbers(text, replace_floats=cfg['replace_floats'])

        return text
