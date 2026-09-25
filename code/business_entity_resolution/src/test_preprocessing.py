"""
Validation script to test the preprocessing module on a small sample of real data.
Prints before/after examples and calculates basic collision statistics.
"""

import pandas as pd
import os
import glob
from pathlib import Path
from . import preprocessing

def run_validation():
    print("--- Preprocessing Validation ---\n")

    # 1. Load a small sample from different sources
    sample_dfs = []
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    INPUT_DIR = PROJECT_ROOT / "dataset" / "dataset" / "test"
    files = glob.glob(str(INPUT_DIR / "*.tsv"))

    for f in files:
        df = pd.read_csv(f, sep='\t', nrows=500)
        df['source_file'] = os.path.basename(f)
        sample_dfs.append(df)

    df_sample = pd.concat(sample_dfs, ignore_index=True)

    print(f"Loaded sample with {len(df_sample)} rows.")

    # 2. Apply preprocessing
    processed_df = preprocessing.preprocess_dataframe(df_sample)

    # 3. Print representative before/after examples
    print("\n--- Example Transformations ---")
    test_cases = [
        ('business_name', 'name_clean'),
        ('business_name', 'name_no_legal_suffix'),
        ('business_address', 'address_clean'),
        ('business_address', 'address_numbers'),
        ('country', 'country_clean')
    ]

    # Try to find interesting examples (with punctuation, suffixes, etc.)
    for raw_col, clean_col in test_cases:
        print(f"\nColumn: {raw_col} -> {clean_col}")
        # Show first 5 examples where they differ
        diffs = processed_df[processed_df[raw_col].astype(str) != processed_df[clean_col].astype(str)]
        if not diffs.empty:
            for idx, row in diffs.head(5).iterrows():
                print(f"  Raw: {row[raw_col]}")
                print(f"  Clean: {row[clean_col]}")
                print("-" * 20)
        else:
            print("  No differences found in sample.")

    # 4. Collision Analysis
    print("\n--- Collision Analysis (Sample) ---")

    # Raw names vs Clean names
    raw_unique = processed_df['business_name'].nunique()
    clean_unique = processed_df['name_clean'].nunique()
    suffix_unique = processed_df['name_no_legal_suffix'].nunique()

    print(f"Unique Raw Names: {raw_unique}")
    print(f"Unique Clean Names: {clean_unique}")
    print(f"Unique No-Suffix Names: {suffix_unique}")

    # Collisions: How many raw names map to the same clean name?
    collision_counts = processed_df.groupby('name_clean')['business_name'].nunique()
    max_collisions = collision_counts.max()
    total_collisions = (collision_counts > 1).sum()

    print(f"Total Clean Names with collisions: {total_collisions}")
    print(f"Max raw names mapping to one clean name: {max_collisions}")

    # 5. Missing value check
    print("\n--- Missing Value Check ---")
    print(f"Missing Raw Names: {processed_df['business_name'].isna().sum()}")
    print(f"Empty Clean Names: {(processed_df['name_clean'] == '').sum()}")
    print(f"Missing Raw Addresses: {processed_df['business_address'].isna().sum()}")
    print(f"Empty Clean Addresses: {(processed_df['address_clean'] == '').sum()}")

if __name__ == "__main__":
    run_validation()
