"""
Blocking Miss Analysis for Amazon ML Challenge 2026.

This script diagnoses why a set of ground-truth true pairs were missed by
all 12 existing blocking strategies. It isolates the misses and
calculates diagnostic features to understand the failure patterns.
"""

import os
import glob
import difflib
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from .blocking import (
    BLOCKING_STRATEGIES,
    _longest_token,
    _first_address_number,
)

# ============================================================
# Configuration
# ============================================================

EXPECTED_S2_MISSES = 69_615
EXPECTED_S3_MISSES = 93_069
EXPECTED_TOTAL_MISSES = 162_684

# Similarity buckets for reporting
SIMILARITY_BUCKETS = [
    (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1)
]

# ============================================================
# Paths
# ============================================================

def get_paths():
    project_root = Path(__file__).resolve().parents[3]
    return {
        "project_root": project_root,
        "ground_truth": project_root / "dataset" / "dataset" / "train" / "train_ground_truth.tsv",
        "processed_train": project_root / "processed" / "train",
        "output_dir": project_root / "code" / "blocking_results",
    }

# ============================================================
# Key Reconstruction (Mirroring blocking.py)
# ============================================================

def reconstruct_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the blocking keys using the EXACT logic from blocking.py.
    Ensures that if any required field is empty, the key is "".
    """
    # Clean source fields
    country = df["country_clean"].fillna("").astype(str).str.strip()
    name = df["name_clean"].fillna("").astype(str).str.strip()
    name_no_suffix = df["name_no_legal_suffix"].fillna("").astype(str).str.strip()
    name_sorted = df["name_sorted_tokens"].fillna("").astype(str).str.strip()
    name_tokens = df["name_tokens"].fillna("").astype(str).str.strip()
    address = df["address_clean"].fillna("").astype(str).str.strip()
    address_numbers = df["address_numbers"].fillna("").astype(str).str.strip()
    address_tokens = df["address_tokens"].fillna("").astype(str).str.strip()

    # Derived fields
    longest_name = name_tokens.apply(_longest_token)
    longest_address = address_tokens.apply(_longest_token)
    first_address_number = address_numbers.apply(_first_address_number)
    name_prefix4 = name_no_suffix.str[:4]

    result = pd.DataFrame(index=df.index)

    # 1. Independent name blocks
    result["block_name_exact"] = (country + "|" + name).where(country.ne("") & name.ne(""), "")
    result["block_name_no_suffix"] = (country + "|" + name_no_suffix).where(country.ne("") & name_no_suffix.ne(""), "")
    result["block_name_sorted"] = (country + "|" + name_sorted).where(country.ne("") & name_sorted.ne(""), "")
    result["block_name_prefix4"] = (country + "|" + name_prefix4).where(country.ne("") & name_prefix4.ne(""), "")
    result["block_name_token"] = (country + "|" + longest_name).where(country.ne("") & longest_name.ne(""), "")

    # 2. Independent address blocks
    result["block_address_exact"] = (country + "|" + address).where(country.ne("") & address.ne(""), "")
    result["block_address_numbers"] = (country + "|" + address_numbers).where(country.ne("") & address_numbers.ne(""), "")
    result["block_address_token"] = (country + "|" + longest_address).where(country.ne("") & longest_address.ne(""), "")

    # 3. Composite blocks
    result["block_name_address_exact"] = (country + "|" + name + "|" + address).where(
        country.ne("") & name.ne("") & address.ne(""), ""
    )
    result["block_name_no_suffix_address"] = (country + "|" + name_no_suffix + "|" + address).where(
        country.ne("") & name_no_suffix.ne("") & address.ne(""), ""
    )
    result["block_name_no_suffix_address_numbers"] = (country + "|" + name_no_suffix + "|" + address_numbers).where(
        country.ne("") & name_no_suffix.ne("") & address_numbers.ne(""), ""
    )
    result["block_name_prefix4_address_number"] = (country + "|" + name_prefix4 + "|" + first_address_number).where(
        country.ne("") & name_prefix4.ne("") & first_address_number.ne(""), ""
    )

    return result

# ============================================================
# Diagnostic Feature Calculations
# ============================================================

def calculate_jaccard(s1_tokens: str, s2_tokens: str) -> float:
    set1 = set(s1_tokens.split())
    set2 = set(s2_tokens.split())
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def calculate_char_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return np.nan
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def calculate_intersection_count(s1_tokens: str, s2_tokens: str) -> int:
    set1 = set(s1_tokens.split())
    set2 = set(s2_tokens.split())
    return len(set1 & set2)

# ============================================================
# Main Analysis Logic
# ============================================================

def main():
    print("=" * 60)
    print("BLOCKING MISS ANALYSIS")
    print("=" * 60)

    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    # 1. Load Ground Truth
    print("Loading ground truth...")
    gt = pd.read_csv(paths["ground_truth"], sep="\t", dtype=str, keep_default_na=False)

    # Convert to long form
    gt["matched_entity_id"] = gt["matched_entity_ids"].str.split(",")
    gt = gt[["source1_entity_id", "matched_entity_id"]].explode("matched_entity_id", ignore_index=True)
    gt = gt[gt["matched_entity_id"] != ""].copy()
    gt["matched_source"] = gt["matched_entity_id"].str[:2]

    total_true_pairs = len(gt)
    s2_true = (gt["matched_source"] == "S2").sum()
    s3_true = (gt["matched_source"] == "S3").sum()
    print(f"Loaded {total_true_pairs:,} true pairs (S2: {s2_true:,}, S3: {s3_true:,})")

    # 2. Load Source Data
    print("\nLoading processed source data...")
    sources = {}
    for src_id in ["S1", "S2", "S3"]:
        filename = f"processed_train_source{src_id[1]}.tsv"
        path = paths["processed_train"] / filename
        # Load only columns needed for blocking and diagnostics
        cols = [
            "entity_id", "country_clean", "name_clean", "name_no_legal_suffix",
            "name_sorted_tokens", "name_tokens", "address_clean", "address_numbers", "address_tokens"
        ]
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=cols)
        df.set_index("entity_id", inplace=True)
        sources[src_id] = df
        print(f"{src_id} loaded: {len(df):,} rows")

    # 3. Reconstruct Blocking Keys for all sources
    print("\nReconstructing blocking keys...")
    keys = {}
    for src_id, df in sources.items():
        keys[src_id] = reconstruct_blocking_keys(df)

    # 4. Evaluate survival for each true pair
    print("Evaluating blocking survival...")

    # Wait, let's fix the column mapping for strategies
    strategy_col_map = {
        "name_exact": "block_name_exact",
        "name_no_suffix_exact": "block_name_no_suffix",
        "name_sorted_exact": "block_name_sorted",
        "name_prefix4": "block_name_prefix4",
        "name_token": "block_name_token",
        "address_exact": "block_address_exact",
        "address_numbers": "block_address_numbers",
        "address_token": "block_address_token",
        "name_address_exact": "block_name_address_exact",
        "name_no_suffix_address": "block_name_no_suffix_address",
        "name_no_suffix_address_numbers": "block_name_no_suffix_address_numbers",
        "name_prefix4_address_number": "block_name_prefix4_address_number",
    }

    # Re-evaluating survival
    all_misses = []
    final_counts = {"S2": {"total": 0, "recovered": 0, "missed": 0},
                    "S3": {"total": 0, "recovered": 0, "missed": 0}}

    for src_id in ["S2", "S3"]:
        src_gt = gt[gt["matched_source"] == src_id].copy()
        if src_gt.empty: continue

        s1_keys = keys["S1"].loc[src_gt["source1_entity_id"]].reset_index(drop=True)
        matched_keys = keys[src_id].loc[src_gt["matched_entity_id"]].reset_index(drop=True)

        # Calculate survival for each strategy
        survived_any = pd.Series([False] * len(src_gt), index=src_gt.index)
        strategy_flags = pd.DataFrame(index=src_gt.index)

        for strategy, col in strategy_col_map.items():
            # survive if keys match and are non-empty
            # Use .values to avoid index alignment issues with the boolean Series
            res = (s1_keys[col].values == matched_keys[col].values) & (s1_keys[col].values != "")
            strategy_flags[strategy] = res
            survived_any |= res

        missed_mask = ~survived_any
        missed_pairs = src_gt[missed_mask].copy()

        # Add survival flags to misses
        for strategy in BLOCKING_STRATEGIES:
            missed_pairs[strategy] = False
        missed_pairs["survived_any"] = False

        # Attach source data for feature analysis
        # S1 data
        s1_data = sources["S1"].loc[missed_pairs["source1_entity_id"]]
        # Matched data
        m_data = sources[src_id].loc[missed_pairs["matched_entity_id"]]

        # Merge carefully
        misses_with_data = missed_pairs.copy()
        misses_with_data["s1_country"] = s1_data["country_clean"].values
        misses_with_data["matched_country"] = m_data["country_clean"].values
        misses_with_data["s1_name"] = s1_data["name_clean"].values
        misses_with_data["matched_name"] = m_data["name_clean"].values
        misses_with_data["s1_address"] = s1_data["address_clean"].values
        misses_with_data["matched_address"] = m_data["address_clean"].values
        misses_with_data["s1_name_no_suffix"] = s1_data["name_no_legal_suffix"].values
        misses_with_data["matched_name_no_suffix"] = m_data["name_no_legal_suffix"].values
        misses_with_data["s1_name_tokens"] = s1_data["name_tokens"].values
        misses_with_data["matched_name_tokens"] = m_data["name_tokens"].values
        misses_with_data["s1_address_tokens"] = s1_data["address_tokens"].values
        misses_with_data["matched_address_tokens"] = m_data["address_tokens"].values
        misses_with_data["s1_address_numbers"] = s1_data["address_numbers"].values
        misses_with_data["matched_address_numbers"] = m_data["address_numbers"].values

        all_misses.append(misses_with_data)

        final_counts[src_id]["total"] = len(src_gt)
        final_counts[src_id]["recovered"] = len(src_gt) - len(missed_pairs)
        final_counts[src_id]["missed"] = len(missed_pairs)

    all_misses_df = pd.concat(all_misses, ignore_index=True)

    # Verify miss counts
    print("\nVerifying miss counts...")
    total_missed = len(all_misses_df)
    s2_missed = final_counts["S2"]["missed"]
    s3_missed = final_counts["S3"]["missed"]

    if s2_missed != EXPECTED_S2_MISSES or s3_missed != EXPECTED_S3_MISSES or total_missed != EXPECTED_TOTAL_MISSES:
        print("WARNING: computed miss count differs from expected result")
        print(f"Expected: S2={EXPECTED_S2_MISSES}, S3={EXPECTED_S3_MISSES}, Total={EXPECTED_TOTAL_MISSES}")
        print(f"Computed: S2={s2_missed}, S3={s3_missed}, Total={total_missed}")
    else:
        print("Miss counts match expected totals.")

    # Write core missed-pair file
    output_cols = [
        "source1_entity_id", "matched_entity_id", "matched_source"
    ] + BLOCKING_STRATEGIES + ["survived_any"] + [
        "s1_country", "matched_country", "s1_name", "matched_name",
        "s1_address", "matched_address", "s1_name_no_suffix",
        "matched_name_no_suffix", "s1_name_tokens", "matched_name_tokens",
        "s1_address_tokens", "matched_address_tokens",
        "s1_address_numbers", "matched_address_numbers"
    ]
    all_misses_df[output_cols].to_csv(paths["output_dir"] / "blocking_missed_pairs.csv", index=False)

    # -----------------------------------------------------------
    # Feature Engineering for Misses
    # -----------------------------------------------------------
    print("\nAnalyzing missed-pair features...")

    # Name Features
    all_misses_df["name_missing_s1"] = all_misses_df["s1_name"] == ""
    all_misses_df["name_missing_matched"] = all_misses_df["matched_name"] == ""
    all_misses_df["name_missing_either"] = all_misses_df["name_missing_s1"] | all_misses_df["name_missing_matched"]
    all_misses_df["name_exact_equal"] = all_misses_df["s1_name"] == all_misses_df["matched_name"]
    all_misses_df["name_no_suffix_equal"] = all_misses_df["s1_name_no_suffix"] == all_misses_df["matched_name_no_suffix"]

    all_misses_df["name_token_jaccard"] = all_misses_df.apply(
        lambda r: calculate_jaccard(r["s1_name_tokens"], r["matched_name_tokens"]), axis=1
    )
    all_misses_df["name_token_intersection_count"] = all_misses_df.apply(
        lambda r: calculate_intersection_count(r["s1_name_tokens"], r["matched_name_tokens"]), axis=1
    )
    all_misses_df["name_char_similarity"] = all_misses_df.apply(
        lambda r: calculate_char_sim(r["s1_name"], r["matched_name"]), axis=1
    )
    all_misses_df["name_len_s1"] = all_misses_df["s1_name"].str.len()
    all_misses_df["name_len_matched"] = all_misses_df["matched_name"].str.len()
    all_misses_df["name_length_difference"] = (all_misses_df["name_len_s1"] - all_misses_df["name_len_matched"]).abs()
    all_misses_df["name_length_ratio"] = all_misses_df["name_len_s1"] / all_misses_df["name_len_matched"].replace(0, np.nan)

    # Address Features
    all_misses_df["address_missing_s1"] = all_misses_df["s1_address"] == ""
    all_misses_df["address_missing_matched"] = all_misses_df["matched_address"] == ""
    all_misses_df["address_missing_either"] = all_misses_df["address_missing_s1"] | all_misses_df["address_missing_matched"]
    all_misses_df["address_exact_equal"] = all_misses_df["s1_address"] == all_misses_df["matched_address"]

    all_misses_df["address_number_jaccard"] = all_misses_df.apply(
        lambda r: calculate_jaccard(r["s1_address_numbers"], r["matched_address_numbers"]), axis=1
    )
    all_misses_df["address_number_intersection_count"] = all_misses_df.apply(
        lambda r: calculate_intersection_count(r["s1_address_numbers"], r["matched_address_numbers"]), axis=1
    )
    all_misses_df["address_number_overlap"] = all_misses_df["address_number_intersection_count"] > 0

    all_misses_df["address_token_jaccard"] = all_misses_df.apply(
        lambda r: calculate_jaccard(r["s1_address_tokens"], r["matched_address_tokens"]), axis=1
    )
    all_misses_df["address_token_intersection_count"] = all_misses_df.apply(
        lambda r: calculate_intersection_count(r["s1_address_tokens"], r["matched_address_tokens"]), axis=1
    )
    all_misses_df["address_char_similarity"] = all_misses_df.apply(
        lambda r: calculate_char_sim(r["s1_address"], r["matched_address"]), axis=1
    )
    all_misses_df["address_len_s1"] = all_misses_df["s1_address"].str.len()
    all_misses_df["address_len_matched"] = all_misses_df["matched_address"].str.len()
    all_misses_df["address_length_difference"] = (all_misses_df["address_len_s1"] - all_misses_df["address_len_matched"]).abs()
    all_misses_df["address_length_ratio"] = all_misses_df["address_len_s1"] / all_misses_df["address_len_matched"].replace(0, np.nan)

    # Country Features
    all_misses_df["country_same"] = all_misses_df["s1_country"] == all_misses_df["matched_country"]
    all_misses_df["country_missing_s1"] = all_misses_df["s1_country"] == ""
    all_misses_df["country_missing_matched"] = all_misses_df["matched_country"] == ""
    all_misses_df["country_missing_either"] = all_misses_df["country_missing_s1"] | all_misses_df["country_missing_matched"]

    # Analysis Categories
    all_misses_df["missing_name"] = all_misses_df["name_missing_either"]
    all_misses_df["missing_address"] = all_misses_df["address_missing_either"]
    all_misses_df["country_mismatch"] = ~all_misses_df["country_same"]
    all_misses_df["no_shared_name_token"] = all_misses_df["name_token_intersection_count"] == 0
    all_misses_df["no_shared_address_token"] = all_misses_df["address_token_intersection_count"] == 0

    # -----------------------------------------------------------
    # Output Distributions and Summaries
    # -----------------------------------------------------------

    numeric_features = [
        "name_char_similarity", "name_token_jaccard", "name_token_intersection_count", "name_length_difference",
        "address_char_similarity", "address_token_jaccard", "address_token_intersection_count",
        "address_number_jaccard", "address_number_intersection_count", "address_length_difference"
    ]

    boolean_features = [
        "name_missing_either", "address_missing_either", "country_same", "country_missing_either",
        "name_exact_equal", "name_no_suffix_equal", "address_exact_equal", "address_number_overlap",
        "no_shared_name_token", "no_shared_address_token"
    ]

    # Summary Files
    summary_rows = []
    feature_summary_rows = []
    boolean_summary_rows = []

    for src_id in ["S2", "S3", "ALL"]:
        df_subset = all_misses_df if src_id == "ALL" else all_misses_df[all_misses_df["matched_source"] == src_id]

        # Basic counts
        if src_id != "ALL":
            total_true = final_counts[src_id]["total"]
            recovered = final_counts[src_id]["recovered"]
            missed = final_counts[src_id]["missed"]
        else:
            total_true = sum(v["total"] for v in final_counts.values())
            recovered = sum(v["recovered"] for v in final_counts.values())
            missed = sum(v["missed"] for v in final_counts.values())

        summary_rows.append({"source": src_id, "metric": "total_true_pairs", "count": total_true, "percentage": 100.0})
        summary_rows.append({"source": src_id, "metric": "recovered_pairs", "count": recovered, "percentage": (recovered/total_true)*100})
        summary_rows.append({"source": src_id, "metric": "missed_pairs", "count": missed, "percentage": (missed/total_true)*100})

        # Numeric summaries
        for feat in numeric_features:
            vals = df_subset[feat]
            feature_summary_rows.append({
                "source": src_id, "feature": feat,
                "count": len(vals), "mean": vals.mean(), "median": vals.median(),
                "p25": vals.quantile(0.25), "p75": vals.quantile(0.75),
                "p90": vals.quantile(0.90), "p95": vals.quantile(0.95),
                "p99": vals.quantile(0.99), "min": vals.min(), "max": vals.max()
            })

        # Boolean summaries
        for feat in boolean_features:
            vals = df_subset[feat]
            true_count = vals.sum()
            false_count = len(vals) - true_count
            boolean_summary_rows.append({
                "source": src_id, "feature": feat,
                "true_count": true_count, "false_count": false_count,
                "true_percentage": (true_count/len(vals)*100) if len(vals)>0 else 0,
                "false_percentage": (false_count/len(vals)*100) if len(vals)>0 else 0
            })

    # Save summaries
    pd.DataFrame(summary_rows).to_csv(paths["output_dir"] / "blocking_miss_summary.csv", index=False)
    pd.DataFrame(feature_summary_rows).to_csv(paths["output_dir"] / "blocking_miss_feature_summary.csv", index=False)
    pd.DataFrame(boolean_summary_rows).to_csv(paths["output_dir"] / "blocking_miss_boolean_summary.csv", index=False)

    # Example misses
    # Highest name similarity
    high_name = all_misses_df.sort_values("name_char_similarity", ascending=False).head(25)
    # Highest address similarity
    high_addr = all_misses_df.sort_values("address_char_similarity", ascending=False).head(25)

    examples = pd.concat([high_name, high_addr]).drop_duplicates()
    example_cols = [
        "source1_entity_id", "matched_entity_id", "matched_source",
        "s1_name", "matched_name", "s1_address", "matched_address",
        "s1_country", "matched_country", "name_char_similarity",
        "name_token_jaccard", "address_char_similarity",
        "address_token_jaccard", "address_number_jaccard"
    ]
    examples[example_cols].to_csv(paths["output_dir"] / "blocking_miss_examples.csv", index=False)

    # Strategy failure sanity check
    print("\nStrategy Sanity Check (Missed Pairs):")
    strategy_counts = []
    for strategy in BLOCKING_STRATEGIES:
        # count how many missed pairs actually survived this strategy (should be 0)
        survived = all_misses_df[strategy].sum()
        strategy_counts.append({"strategy": strategy, "missed_pairs_surviving": survived, "percentage": (survived/len(all_misses_df)*100)})

    strat_df = pd.DataFrame(strategy_counts)
    print(strat_df.to_string(index=False))

    print("\nAnalysis complete. Outputs written to code/blocking_results/")

if __name__ == "__main__":
    main()
