"""
Configuration for the Amazon ML Challenge 2026 preprocessing pipeline.
Centralizes all normalization lists and constants to ensure consistency across train/test.
"""

import re

# --- Business Name Normalization ---
# Legal suffixes to remove for 'name_no_legal_suffix'.
# Included common suffixes for US, India, and France.
LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "pvt",
    "sarl",
    "sa",
    "sas",
    "eurl",
    "sc",
    "snc"
}

# Mapping for common character replacements in names
NAME_REPLACEMENTS = {
    "&": " and ",
    "@": " at "
}

# --- Address Normalization ---
# Common address abbreviations for standardization.
ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "lane": "ln",
    "drive": "dr",
    "apartment": "apt",
    "suite": "ste",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "terrace": "ter",
    "way": "wy"
}

# Regex to extract numbers from addresses (captures sequences of digits)
ADDRESS_NUMBER_REGEX = re.compile(r'\d+')

# --- General Settings ---
UNICODE_NORMALIZATION = "NFKC"
MIN_TOKEN_LENGTH = 1  # Minimum length for a token to be kept
