"""
Preprocessing module for the Amazon ML Challenge 2026.
Provides reusable functions for normalizing business names, addresses, and countries.
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Set, Optional
from . import config

def normalize_text(text: Optional[str]) -> str:
    """
    Basic text normalization: handles NaNs, Unicode, case, and whitespace.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    # Cast to string and Unicode normalize
    text = str(text)
    text = unicodedata.normalize(config.UNICODE_NORMALIZATION, text)

    # Case normalization
    text = text.casefold()

    # Whitespace normalization: strip and collapse repeated spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def normalize_business_name(text: Optional[str]) -> str:
    """
    Unicode-safe normalization for business names.

    Preserves non-Latin scripts while normalizing:
    - Unicode representation
    - case
    - &, @ replacements
    - punctuation
    - whitespace
    """
    text = normalize_text(text)

    if not text:
        return ""

    # Character replacements
    for old, new in config.NAME_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Keep Unicode letters/numbers.
    # Replace punctuation/symbols with spaces.
    cleaned_chars = []

    for ch in text:
        if ch.isalnum() or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_legal_suffix(text: str) -> str:
    """
    Removes legal suffixes only from the end of a normalized
    business name.
    """
    tokens = text.split()

    while tokens and tokens[-1] in config.LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens).strip()

def tokenize_text(text: str) -> List[str]:
    """
    Splits text into tokens, filtering by minimum length.
    """
    tokens = text.split()
    return [t for t in tokens if len(t) >= config.MIN_TOKEN_LENGTH]

def normalize_address(text: Optional[str]) -> str:
    """
    Specific normalization for business addresses.

    Preserves Unicode letters and numbers while normalizing:
    - case
    - common address abbreviations
    - punctuation
    - whitespace
    """
    text = normalize_text(text)

    if not text:
        return ""

    # Address abbreviation normalization
    tokens = text.split()
    normalized_tokens = [
        config.ADDRESS_ABBREVIATIONS.get(t, t)
        for t in tokens
    ]
    text = " ".join(normalized_tokens)

    # Keep Unicode letters/numbers.
    # Replace punctuation/symbols with spaces.
    cleaned_chars = []

    for ch in text:
        if ch.isalnum() or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_address_numbers(text: Optional[str]) -> str:
    """
    Extracts all digit sequences from an address as a space-separated string.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    matches = config.ADDRESS_NUMBER_REGEX.findall(str(text))
    return " ".join(matches)

def normalize_country(text: Optional[str]) -> str:
    """
    Consistent country normalization.
    """
    text = normalize_text(text)
    # Simple mapping if needed, otherwise just lowercase/stripped
    # Based on EDA: US, India, France
    if text == "united states":
        return "us"
    if text == "usa":
        return "us"
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to a DataFrame.
    Creates new columns while preserving originals.
    """
    # Work on a shallow copy to avoid SettingWithCopyWarning if df is a slice
    df = df.copy()

    # 1. Country Normalization
    df['country_clean'] = df['country'].apply(normalize_country)

    # 2. Business Name Pipeline
    df['name_clean'] = df['business_name'].apply(normalize_business_name)
    df['name_no_legal_suffix'] = df['name_clean'].apply(remove_legal_suffix)
    df['name_tokens'] = df['name_clean'].apply(lambda x: " ".join(tokenize_text(x)))

    # Sorted tokens for easier comparison later
    df['name_sorted_tokens'] = df['name_clean'].apply(
        lambda x: " ".join(sorted(tokenize_text(x)))
    )

    # 3. Address Pipeline
    df['address_clean'] = df['business_address'].apply(normalize_address)
    df['address_numbers'] = df['business_address'].apply(extract_address_numbers)
    df['address_tokens'] = df['address_clean'].apply(lambda x: " ".join(tokenize_text(x)))

    # Sorted address tokens
    df['address_sorted_tokens'] = df['address_clean'].apply(
        lambda x: " ".join(sorted(tokenize_text(x)))
    )

    return df
