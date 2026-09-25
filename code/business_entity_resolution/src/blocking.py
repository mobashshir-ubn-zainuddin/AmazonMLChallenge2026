"""
Blocking / candidate-generation utilities for Amazon ML Challenge 2026.

Blocking creates inexpensive keys that group records likely to refer
to the same business.

The strategy deliberately uses multiple blocking routes and takes
their UNION so that recall is protected.

No external data is used.
"""

from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------
# Blocking strategies
# ---------------------------------------------------------------------

BLOCKING_STRATEGIES = [
    # Independent name blocks
    "name_exact",
    "name_no_suffix_exact",
    "name_sorted_exact",
    "name_prefix4",
    "name_token",

    # Independent address blocks
    "address_exact",
    "address_numbers",
    "address_token",

    # Composite name + address blocks
    "name_address_exact",
    "name_no_suffix_address",
    "name_no_suffix_address_numbers",
    "name_prefix4_address_number",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clean_series(series: pd.Series) -> pd.Series:
    """Convert missing values to empty strings."""
    return series.fillna("").astype(str).str.strip()


def _longest_token(value: str) -> str:
    """Return the longest token from a whitespace-separated string."""

    tokens = value.split()

    if not tokens:
        return ""

    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )

    for token in tokens:
        if len(token) >= 4:
            return token

    return tokens[0]


def _first_address_number(value: str) -> str:
    """Return the first address number from a space-separated number list."""

    tokens = value.split()

    if not tokens:
        return ""

    return tokens[0]


# ---------------------------------------------------------------------
# Create blocking keys
# ---------------------------------------------------------------------

def create_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all blocking keys for a preprocessed DataFrame.

    Required columns:

        country_clean
        name_clean
        name_no_legal_suffix
        name_sorted_tokens
        name_tokens
        address_clean
        address_numbers
        address_tokens
    """

    required_columns = [
        "country_clean",
        "name_clean",
        "name_no_legal_suffix",
        "name_sorted_tokens",
        "name_tokens",
        "address_clean",
        "address_numbers",
        "address_tokens",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required preprocessed columns: "
            + ", ".join(missing)
        )

    result = df.copy()

    # -------------------------------------------------------------
    # Clean source fields
    # -------------------------------------------------------------

    country = _clean_series(result["country_clean"])

    name = _clean_series(result["name_clean"])

    name_no_suffix = _clean_series(
        result["name_no_legal_suffix"]
    )

    name_sorted = _clean_series(
        result["name_sorted_tokens"]
    )

    name_tokens = _clean_series(
        result["name_tokens"]
    )

    address = _clean_series(
        result["address_clean"]
    )

    address_numbers = _clean_series(
        result["address_numbers"]
    )

    address_tokens = _clean_series(
        result["address_tokens"]
    )

    # -------------------------------------------------------------
    # Derived fields
    # -------------------------------------------------------------

    longest_name = name_tokens.apply(
        _longest_token
    )

    longest_address = address_tokens.apply(
        _longest_token
    )

    first_address_number = address_numbers.apply(
        _first_address_number
    )

    name_prefix4 = name_no_suffix.str[:4]

    # -------------------------------------------------------------
    # 1. Independent name blocks
    # -------------------------------------------------------------

    # A blocking key is valid only when all fields required by
    # that strategy are non-empty.
    result["block_name_exact"] = (
        country + "|" + name
    ).where(
        country.ne("") & name.ne(""),
        ""
    )

    result["block_name_no_suffix"] = (
        country + "|" + name_no_suffix
    ).where(
        country.ne("") & name_no_suffix.ne(""),
        ""
    )

    result["block_name_sorted"] = (
        country + "|" + name_sorted
    ).where(
        country.ne("") & name_sorted.ne(""),
        ""
    )

    result["block_name_prefix4"] = (
        country + "|" + name_prefix4
    ).where(
        country.ne("") & name_prefix4.ne(""),
        ""
    )

    result["block_name_token"] = (
        country + "|" + longest_name
    ).where(
        country.ne("") & longest_name.ne(""),
        ""
    )

    # -------------------------------------------------------------
    # 2. Independent address blocks
    # -------------------------------------------------------------

    result["block_address_exact"] = (
        country + "|" + address
    ).where(
        country.ne("") & address.ne(""),
        ""
    )

    result["block_address_numbers"] = (
        country + "|" + address_numbers
    ).where(
        country.ne("") & address_numbers.ne(""),
        ""
    )

    result["block_address_token"] = (
        country + "|" + longest_address
    ).where(
        country.ne("") & longest_address.ne(""),
        ""
    )

    # -------------------------------------------------------------
    # 3. Composite NAME + ADDRESS blocks
    # -------------------------------------------------------------

    # Exact normalized name + exact normalized address
    result["block_name_address_exact"] = (
        country
        + "|"
        + name
        + "|"
        + address
    ).where(
        country.ne("")
        & name.ne("")
        & address.ne(""),
        ""
    )

    # Name after legal-suffix removal + exact address
    result["block_name_no_suffix_address"] = (
        country
        + "|"
        + name_no_suffix
        + "|"
        + address
    ).where(
        country.ne("")
        & name_no_suffix.ne("")
        & address.ne(""),
        ""
    )

    # Name after legal-suffix removal + address numbers
    result["block_name_no_suffix_address_numbers"] = (
        country
        + "|"
        + name_no_suffix
        + "|"
        + address_numbers
    ).where(
        country.ne("")
        & name_no_suffix.ne("")
        & address_numbers.ne(""),
        ""
    )

    # First four characters of name + first address number
    result["block_name_prefix4_address_number"] = (
        country
        + "|"
        + name_prefix4
        + "|"
        + first_address_number
    ).where(
        country.ne("")
        & name_prefix4.ne("")
        & first_address_number.ne(""),
        ""
    )

    return result


# ---------------------------------------------------------------------
# Strategy → column mapping
# ---------------------------------------------------------------------

def get_block_column(strategy: str) -> str:

    mapping = {
        "name_exact":
            "block_name_exact",

        "name_no_suffix_exact":
            "block_name_no_suffix",

        "name_sorted_exact":
            "block_name_sorted",

        "name_prefix4":
            "block_name_prefix4",

        "name_token":
            "block_name_token",

        "address_exact":
            "block_address_exact",

        "address_numbers":
            "block_address_numbers",

        "address_token":
            "block_address_token",

        "name_address_exact":
            "block_name_address_exact",

        "name_no_suffix_address":
            "block_name_no_suffix_address",

        "name_no_suffix_address_numbers":
            "block_name_no_suffix_address_numbers",

        "name_prefix4_address_number":
            "block_name_prefix4_address_number",
    }

    if strategy not in mapping:
        raise ValueError(
            f"Unknown blocking strategy: {strategy}"
        )

    return mapping[strategy]


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def blocking_statistics(
    df: pd.DataFrame,
    strategy: str
) -> Dict[str, float]:

    column = get_block_column(strategy)

    if column not in df.columns:
        raise ValueError(
            f"Blocking column '{column}' not found."
        )

    values = (
        df[column]
        .fillna("")
        .astype(str)
    )

    # Empty keys are not valid blocks.
    values = values[values != ""]

    if values.empty:
        return {
            "strategy": strategy,
            "unique_blocks": 0,
            "max_block_size": 0,
            "mean_block_size": 0.0,
            "median_block_size": 0.0,
        }

    counts = values.value_counts()

    return {
        "strategy": strategy,
        "unique_blocks": int(len(counts)),
        "max_block_size": int(counts.max()),
        "mean_block_size": float(counts.mean()),
        "median_block_size": float(counts.median()),
    }


def all_blocking_statistics(
    df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for strategy in BLOCKING_STRATEGIES:
        rows.append(
            blocking_statistics(
                df,
                strategy
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pair-level blocking
# ---------------------------------------------------------------------

def pair_survives_blocking(
    source1_row: pd.Series,
    candidate_row: pd.Series,
    strategies: List[str] = None,
) -> Dict[str, bool]:

    if strategies is None:
        strategies = BLOCKING_STRATEGIES

    results = {}

    for strategy in strategies:

        column = get_block_column(strategy)

        key1 = source1_row.get(column, "")
        key2 = candidate_row.get(column, "")

        if pd.isna(key1):
            key1 = ""

        if pd.isna(key2):
            key2 = ""

        key1 = str(key1)
        key2 = str(key2)

        results[strategy] = (
            key1 != ""
            and key2 != ""
            and key1 == key2
        )

    return results


def pair_survives_any_block(
    source1_row: pd.Series,
    candidate_row: pd.Series,
    strategies: List[str] = None,
) -> bool:

    results = pair_survives_blocking(
        source1_row,
        candidate_row,
        strategies,
    )

    return any(results.values())