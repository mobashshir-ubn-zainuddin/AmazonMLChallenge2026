"""
Text normalization v2 for names and addresses.

Fixes two bugs in the v1 preprocessing (src/preprocessing.py):
  * v1 kept only str.isalnum() characters, which deleted Indic vowel signs
    (Unicode category M*) and shattered words: "प्राइवेट" -> "प र इव ट".
  * v1 mapped address abbreviations before removing punctuation, so "Drive,"
    never became "dr".

Adds normalizations for noise seen in the training ground truth:
  accents, leet/OCR digit-letter swaps, legal suffixes anywhere in the name,
  domain-style names, dotted initials ("l.l.c."), glued numbers ("no15abc"),
  zero-padded numbers, state full-name/abbreviation variants, and native-script
  (Hindi/Kannada/Tamil/...) names via dictionaries learned from TRAIN labels only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------

# Everything that is not a word char, whitespace, Latin combining mark or an
# Indic-block character (which includes the vowel signs v1 destroyed) becomes a
# space. "_" is treated as punctuation.
_PUNCT_RE = re.compile(r"[^\w\sऀ-෿]|_")
_LATIN_MARKS_RE = re.compile(r"[̀-ͯ]")
_WS_RE = re.compile(r"\s+")
_DIGIT_LETTER_RE = re.compile(r"(\d)([^\W\d_])")
_LETTER_DIGIT_RE = re.compile(r"([^\W\d_])(\d)")
_ASCII_LETTER_RE = re.compile(r"[a-z]")

LEGAL_TOKENS = {
    "inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation",
    "co", "company", "plc", "pvt", "private", "llp", "lp", "sarl", "sa",
    "sas", "eurl", "snc", "sc", "pllc", "pc", "ms", "the", "and",
}
DOMAIN_TOKENS = {"www", "http", "https"}
TLD_TOKENS = ["com", "in", "net", "org", "co", "biz", "info", "fr", "us", "io"]

_LEET = str.maketrans({"0": "o", "1": "i", "l": "i", "5": "s", "3": "e", "4": "a", "7": "t", "$": "s"})

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "ts": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "new dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "andaman and nicobar islands": "an", "lakshadweep": "ld",
}
ADDRESS_ABBREV = {
    "street": "st", "str": "st", "road": "rd", "drive": "dr", "avenue": "ave",
    "av": "ave", "boulevard": "blvd", "bd": "blvd", "lane": "ln", "court": "ct",
    "place": "pl", "square": "sq", "terrace": "ter", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "trail": "trl", "apartment": "apt",
    "apartments": "apt", "suite": "ste", "floor": "flr", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt",
    "saint": "st", "fort": "ft", "first": "1st", "second": "2nd",
    "third": "3rd", "fourth": "4th", "fifth": "5th", "near": "nr",
    "opposite": "opp", "sector": "sec", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "mumbai": "mumbai", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai",
}
ADDRESS_DROP = {"null", "none", "nan", "township", "city", "no", "nos", "hno",
                "h", "d", "dno", "ho", "po", "p", "o", "the", "of", "and"}

_PHRASES = {}
for _full, _abbr in list(US_STATES.items()) + list(IN_STATES.items()):
    if " " in _full:
        _PHRASES[_full] = _abbr
_PHRASE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _PHRASES), key=len, reverse=True)) + r")\b")
_SINGLE_STATE = {k: v for k, v in list(US_STATES.items()) + list(IN_STATES.items()) if " " not in k}


# ---------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------

def _base_clean(text: str) -> str:
    """casefold + NFKC + Latin accent removal + punctuation -> space."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text).casefold()
    if not s.isascii():
        s = _LATIN_MARKS_RE.sub("", unicodedata.normalize("NFKD", s))
        s = unicodedata.normalize("NFC", s)
    return s


def _punct_to_space(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", s)).strip()


def _join_initials(tokens: List[str]) -> List[str]:
    """Join runs of single Latin letters: 'l l c' -> 'llc', 'm s' -> 'ms'."""
    out: List[str] = []
    run: List[str] = []
    for t in tokens:
        if len(t) == 1 and "a" <= t <= "z":
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def is_non_latin(token: str) -> bool:
    """True for tokens containing letters outside ASCII (native scripts)."""
    return not token.isascii() and any(ch.isalpha() and ord(ch) > 0x24F for ch in token)


def has_latin_letter(text: str) -> bool:
    return bool(_ASCII_LETTER_RE.search(_base_clean(text)))


# ---------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------

def name_tokens(raw: str, translit: Optional[Dict[str, str]] = None) -> List[str]:
    s = _base_clean(raw)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _punct_to_space(s)
    tokens = _join_initials(s.split())
    if translit:
        tokens = [translit.get(t, t) if not t.isascii() else t for t in tokens]
        tokens = " ".join(tokens).split()
    return tokens


def normalize_name(raw: str, translit: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """
    Returns (name_clean, name_key).

    name_clean: readable normalized name (keeps legal words).
    name_key:   retrieval key: legal/domain tokens removed anywhere, leet folded.
    """
    tokens = name_tokens(raw, translit)
    clean = " ".join(tokens)
    low = raw.casefold()
    key = []
    for i, t in enumerate(tokens):
        if t in LEGAL_TOKENS or t in DOMAIN_TOKENS:
            continue
        if t in TLD_TOKENS and i > 0 and ("." + t) in low:
            continue
        if t.isascii() and not t.isdigit():
            t = t.translate(_LEET)
        key.append(t)
    if not key:
        key = [t.translate(_LEET) if t.isascii() else t for t in tokens]
    return clean, " ".join(key)


# ---------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------

def normalize_address(
    raw: str,
    translit: Optional[Dict[str, str]] = None,
    seg_translit: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    """Returns (address_clean, numbers) — numbers are zero-stripped digit tokens in order."""
    s = _base_clean(raw)
    if not s:
        return "", []
    if seg_translit and not s.isascii():
        segs = []
        for seg in s.split(","):
            key = _punct_to_space(seg)
            segs.append(seg_translit.get(key, seg))
        s = ",".join(segs)
    s = _DIGIT_LETTER_RE.sub(r"\1 \2", s)
    s = _LETTER_DIGIT_RE.sub(r"\1 \2", s)
    s = _punct_to_space(s)
    if translit and not s.isascii():
        s = " ".join(translit.get(t, t) if not t.isascii() else t for t in s.split())
    s = _PHRASE_RE.sub(lambda m: _PHRASES[m.group(1)], s)

    tokens: List[str] = []
    nums: List[str] = []
    for t in s.split():
        if t.isdigit():
            try:
                t = str(int(t))
            except ValueError:
                pass
            tokens.append(t)
            if t not in nums:
                nums.append(t)
            continue
        if len(t) > 8 and t.endswith("township"):
            t = t[:-8]
        if t in ADDRESS_DROP:
            continue
        t = _SINGLE_STATE.get(t, t)
        t = ADDRESS_ABBREV.get(t, t)
        tokens.append(t)
    return " ".join(tokens), nums


def numbers_to_array(num_lists: Sequence[List[str]], width: int = 6) -> np.ndarray:
    """Encode each record's address numbers as up to `width` int64 values (-1 pad)."""
    out = np.full((len(num_lists), width), -1, dtype=np.int64)
    for i, nums in enumerate(num_lists):
        for j, n in enumerate(nums[:width]):
            out[i, j] = int(n[-18:])
    return out


# ---------------------------------------------------------------------
# Learning native-script dictionaries from TRAIN ground truth only
# ---------------------------------------------------------------------

def learn_name_translit(
    s1_names: Sequence[str],
    match_names: Sequence[str],
    min_count: int = 2,
    min_share: float = 0.5,
) -> Dict[str, str]:
    """
    Align native-script match names with their Latin S1 name token-by-token
    (only when token counts agree) and keep confident (native -> latin) pairs.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    for s1_raw, m_raw in zip(s1_names, match_names):
        m_tok = name_tokens(m_raw)
        if not m_tok or not any(is_non_latin(t) for t in m_tok):
            continue
        s_tok = name_tokens(s1_raw)
        if len(s_tok) != len(m_tok):
            continue
        for a, b in zip(m_tok, s_tok):
            if is_non_latin(a) and b.isascii():
                counts[a][b] += 1
    out = {}
    for nat, c in counts.items():
        total = sum(c.values())
        lat, n = c.most_common(1)[0]
        if n >= min_count and n / total >= min_share:
            out[nat] = lat
    return out


def learn_segment_translit(
    s1_addrs: Sequence[str],
    match_addrs: Sequence[str],
    min_count: int = 3,
    min_share: float = 0.3,
) -> Dict[str, str]:
    """
    Learn native-script address segments (mostly state/city names) -> Latin
    segment, by counting which S1 comma-segment is absent from the match's
    Latin text whenever a native segment appears.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    seen: Counter = Counter()
    for s1_raw, m_raw in zip(s1_addrs, match_addrs):
        m = _base_clean(m_raw)
        if not m or m.isascii():
            continue
        m_segs = [_punct_to_space(x) for x in m.split(",")]
        native = [x for x in m_segs if x and any(is_non_latin(t) for t in x.split())]
        if not native:
            continue
        m_text = " " + " ".join(m_segs) + " "
        s1_segs = [_punct_to_space(x) for x in _base_clean(s1_raw).split(",")]
        absent = {x for x in s1_segs if x and not x.isdigit() and (" " + x + " ") not in m_text}
        for nat in native:
            seen[nat] += 1
            for lat in absent:
                counts[nat][lat] += 1
    out = {}
    for nat, c in counts.items():
        lat, n = c.most_common(1)[0]
        # share = fraction of occurrences of this native segment where the
        # candidate Latin segment was the one missing from the match
        if n >= min_count and n / seen[nat] >= min_share:
            out[nat] = lat
    return out
