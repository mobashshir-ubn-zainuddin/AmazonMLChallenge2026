"""
Blocking Candidate Size Analysis for Amazon ML Challenge 2026.

This script measures the volume of candidates produced by each existing
blocking strategy on the full training data.

Individual strategy statistics use block-key frequency maps.
UNION statistics use compact block indexes over integer row positions
to avoid materializing candidate pairs or Cartesian products.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .blocking import (
    BLOCKING_STRATEGIES,
    _longest_token,
    _first_address_number,
)

# ============================================================
# Configuration
# ============================================================

# Sanity check counts
S1_EXPECTED_COUNT = 2_206_821
S2_EXPECTED_COUNT = 5_034_616
S3_EXPECTED_COUNT = 5_285_603

# Distribution buckets for optional distribution analysis
DISTRIBUTION_BUCKETS = [
    (0, 1), (1, 2), (2, 5), (5, 10), (10, 25), (25, 50),
    (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 5000), (5000, float('inf'))
]

# ============================================================
# Paths
# ============================================================

def get_paths():
    project_root = Path(__file__).resolve().parents[3]
    return {
        "project_root": project_root,
        "processed_train": project_root / "processed" / "train",
        "output_dir": project_root / "code" / "blocking_results",
    }

# ============================================================
# Key Reconstruction (Mirroring blocking.py exactly)
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
# Analysis Core
# ============================================================

def get_candidate_stats(counts: pd.Series) -> Dict[str, Any]:
    """Computes distribution statistics for a series of candidate counts."""
    return {
        "total_s1_entities": len(counts),
        "zero_candidate_s1": (counts == 0).sum(),
        "nonzero_candidate_s1": (counts > 0).sum(),
        "total_candidate_pairs": counts.sum(),
        "mean_candidates": counts.mean(),
        "median_candidates": counts.median(),
        "p75_candidates": counts.quantile(0.75),
        "p90_candidates": counts.quantile(0.90),
        "p95_candidates": counts.quantile(0.95),
        "p99_candidates": counts.quantile(0.99),
        "p999_candidates": counts.quantile(0.999),
        "max_candidates": counts.max(),
    }

def main():
    print("=" * 60)
    print("BLOCKING CANDIDATE SIZE ANALYSIS")
    print("=" * 60)

    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Training Data
    print("\nLoading training data...")
    sources_data = {}
    for src_id in ["S1", "S2", "S3"]:
        filename = f"processed_train_source{src_id[1]}.tsv"
        path = paths["processed_train"] / filename
        cols = [
            "entity_id", "country_clean", "name_clean", "name_no_legal_suffix",
            "name_sorted_tokens", "name_tokens", "address_clean", "address_numbers", "address_tokens"
        ]
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=cols)
        df.set_index("entity_id", inplace=True)
        sources_data[src_id] = df
        print(f"{src_id} loaded: {len(df):,} rows")

    # Sanity Checks
    assert len(sources_data["S1"]) == S1_EXPECTED_COUNT, (
        f"S1 row count mismatch: "
        f"{len(sources_data['S1']):,} != {S1_EXPECTED_COUNT:,}"
    )

    assert len(sources_data["S2"]) == S2_EXPECTED_COUNT, (
        f"S2 row count mismatch: "
        f"{len(sources_data['S2']):,} != {S2_EXPECTED_COUNT:,}"
    )

    assert len(sources_data["S3"]) == S3_EXPECTED_COUNT, (
        f"S3 row count mismatch: "
        f"{len(sources_data['S3']):,} != {S3_EXPECTED_COUNT:,}"
    )

    # 2. Reconstruct Blocking Keys
    print("\nReconstructing blocking keys...")
    source_keys = {}
    for src_id, df in sources_data.items():
        source_keys[src_id] = reconstruct_blocking_keys(df)

    # Mapping strategy names to reconstructed column names
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

    # 3. Analyze candidate counts
    all_strategy_stats = []

    # We analyze S1 -> S2 and S1 -> S3
    for target_id in ["S2", "S3"]:
        print(f"\nAnalyzing S1 -> {target_id} candidates...")

        # Pre-compute counts for every strategy in target source
        target_strategy_counts = {} # strategy -> {block_key: count}
        for strategy, col in strategy_col_map.items():
            keys_series = source_keys[target_id][col]
            # Exclude empty keys
            valid_keys = keys_series[keys_series != ""]
            target_strategy_counts[strategy] = valid_keys.value_counts()

        # For each strategy, calculate candidate counts for all S1
        for strategy, col in strategy_col_map.items():
            print(f"  Strategy: {strategy}")
            s1_keys = source_keys["S1"][col]

            # Map S1 keys to the count of matches in the target source
            counts = s1_keys.map(target_strategy_counts[strategy]).fillna(0).astype(int)

            stats = get_candidate_stats(counts)
            stats["source"] = target_id
            stats["strategy"] = strategy
            all_strategy_stats.append(stats)

    # 4. Union Analysis
    print("\nComputing UNION candidate statistics...")

    def build_compact_block_index(keys_series: pd.Series):
        """
        Build a compact inverted index for one blocking strategy.

        The index maps each non-empty block key to a contiguous range
        in a sorted integer row-position array.

        Returns:
            block_ranges:
                dict mapping block_key -> (start, end)

            sorted_positions:
                numpy array containing target row positions sorted
                according to block key.

        Entity IDs are NOT stored in this index.
        Empty block keys are excluded.
        """

        keys = keys_series.to_numpy(dtype=object)

        valid_mask = keys != ""
        valid_positions = np.flatnonzero(valid_mask)

        if len(valid_positions) == 0:
            return {}, np.empty(0, dtype=np.int32)

        valid_keys = keys[valid_mask]

        order = np.argsort(valid_keys, kind="stable")

        sorted_keys = valid_keys[order]
        sorted_positions = valid_positions[order].astype(
            np.int32,
            copy=False,
        )

        block_ranges = {}

        start = 0
        n = len(sorted_keys)

        while start < n:
            key = sorted_keys[start]

            end = start + 1

            while end < n and sorted_keys[end] == key:
                end += 1

            block_ranges[key] = (start, end)

            start = end

        return block_ranges, sorted_positions

    def compute_union_counts(target_id: str):
        """
        Compute exact S1 -> target UNION candidate counts.

        Each target entity is represented by an integer row position
        rather than its entity ID string.

        For each S1 entity, candidates retrieved by all 12 blocking
        strategies are unioned exactly using a reusable `seen` array.

        No candidate-pair dataframe and no Python set of entity IDs
        is created.
        """

        s1_keys_df = source_keys["S1"]
        target_keys_df = source_keys[target_id]

        n_s1 = len(s1_keys_df)
        n_target = len(target_keys_df)

        union_counts = np.zeros(
            n_s1,
            dtype=np.int32,
        )

        # Build one compact index per strategy.
        strategy_indexes = {}

        for strategy, col in strategy_col_map.items():
            print(
                f"  Building compact UNION index: "
                f"{target_id} / {strategy}"
            )

            block_ranges, sorted_positions = build_compact_block_index(
                target_keys_df[col]
            )

            strategy_indexes[strategy] = (
                block_ranges,
                sorted_positions,
            )

        # Reusable marker array.
        #
        # For the current S1 row i:
        #
        #     seen[target_row] == marker
        #
        # means that target_row has already been counted.
        seen = np.zeros(
            n_target,
            dtype=np.int32,
        )

        # Convert S1 blocking-key columns to numpy arrays for fast
        # positional access during the 2.2M-row loop.
        s1_key_arrays = {
            strategy: s1_keys_df[col].to_numpy(dtype=object)
            for strategy, col in strategy_col_map.items()
        }

        for i in range(n_s1):
            marker = i + 1
            candidate_count = 0

            for strategy in BLOCKING_STRATEGIES:
                key = s1_key_arrays[strategy][i]

                if key == "":
                    continue

                block_ranges, sorted_positions = strategy_indexes[
                    strategy
                ]

                match = block_ranges.get(key)

                if match is None:
                    continue

                start, end = match

                target_rows = sorted_positions[start:end]

                if len(target_rows) == 0:
                    continue

                already_seen = seen[target_rows] == marker

                new_rows = target_rows[~already_seen]

                if len(new_rows) > 0:
                    seen[new_rows] = marker
                    candidate_count += len(new_rows)

            union_counts[i] = candidate_count

        return pd.Series(
            union_counts,
            index=s1_keys_df.index,
        )

    # S1 -> S2 Union
    s2_union_counts = compute_union_counts("S2")
    # S1 -> S3 Union
    s3_union_counts = compute_union_counts("S3")

    # S1 -> S2+S3 Union
    # Use the same compact integer-row-position UNION logic,
    # but treat S2 and S3 as one combined target population.

    combined_target_keys = pd.concat(
        [
            source_keys["S2"].reset_index(drop=True),
            source_keys["S3"].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    def compute_combined_union_counts():
        """
        Compute exact S1 -> (S2 + S3) UNION candidate counts.

        S2 and S3 rows are represented by unique integer positions
        in the combined target table.

        The first S2 row has combined position 0.
        The last S2 row has its corresponding position.
        S3 rows continue immediately after S2 rows.

        Entity IDs are never required for the UNION computation.
        """

        s1_keys_df = source_keys["S1"]

        n_s1 = len(s1_keys_df)
        n_target = len(combined_target_keys)

        union_counts = np.zeros(
            n_s1,
            dtype=np.int32,
        )

        strategy_indexes = {}

        for strategy, col in strategy_col_map.items():
            print(
                f"  Building compact combined UNION index: "
                f"{strategy}"
            )

            block_ranges, sorted_positions = build_compact_block_index(
                combined_target_keys[col]
            )

            strategy_indexes[strategy] = (
                block_ranges,
                sorted_positions,
            )

        seen = np.zeros(
            n_target,
            dtype=np.int32,
        )

        s1_key_arrays = {
            strategy: s1_keys_df[col].to_numpy(dtype=object)
            for strategy, col in strategy_col_map.items()
        }

        for i in range(n_s1):
            marker = i + 1
            candidate_count = 0

            for strategy in BLOCKING_STRATEGIES:
                key = s1_key_arrays[strategy][i]

                if key == "":
                    continue

                block_ranges, sorted_positions = strategy_indexes[
                    strategy
                ]

                match = block_ranges.get(key)

                if match is None:
                    continue

                start, end = match

                target_rows = sorted_positions[start:end]

                if len(target_rows) == 0:
                    continue

                already_seen = seen[target_rows] == marker

                new_rows = target_rows[~already_seen]

                if len(new_rows) > 0:
                    seen[new_rows] = marker
                    candidate_count += len(new_rows)

            union_counts[i] = candidate_count

        return pd.Series(
            union_counts,
            index=s1_keys_df.index,
        )

    s2_s3_union_counts = compute_combined_union_counts()

    # Calculate Union Stats
    union_stats = []
    union_data = {
        "S2_UNION": s2_union_counts,
        "S3_UNION": s3_union_counts,
        "S2_S3_UNION": s2_s3_union_counts
    }

    for label, counts in union_data.items():
        stats = get_candidate_stats(counts)
        stats["source"] = label
        union_stats.append(stats)

    # 5. Per-Strategy Contribution
    # Contribution = how many unique candidates does this strategy add to the union?
    # This is expensive. We'll calculate it for S2 UNION as an example or a simplified version.
    # To be strictly correct and efficient:
    # For each strategy, total_candidates = sum of all counts for that strategy.
    # Unique contribution is hard without the full pair list.
    # We will report the total candidate pairs per strategy.

    contribution_rows = []
    for target_id in ["S2", "S3"]:
        for strategy, col in strategy_col_map.items():
            total_pairs = source_keys["S1"][col].map(
                source_keys[target_id][col].value_counts()
            ).fillna(0).sum()
            contribution_rows.append({
                "source": target_id,
                "strategy": strategy,
                "total_candidate_pairs": total_pairs
            })

    # 6. Write Results
    print("\nWriting results...")

    # Summary CSV
    pd.DataFrame(all_strategy_stats).to_csv(paths["output_dir"] / "blocking_candidate_size_summary.csv", index=False)
    # Union Summary CSV
    pd.DataFrame(union_stats).to_csv(paths["output_dir"] / "blocking_union_size_summary.csv", index=False)
    # Volume CSV
    pd.DataFrame(contribution_rows).to_csv(paths["output_dir"] / "blocking_strategy_candidate_volume.csv", index=False)

    # Examples CSV (Top 100 for Union S2_S3)
    top_s1 = s2_s3_union_counts.sort_values(ascending=False).head(100)
    example_data = []
    for s1_id, count in top_s1.items():
        example_data.append({
            "source1_entity_id": s1_id,
            "source": "S2_S3_UNION",
            "candidate_count": count
        })
    pd.DataFrame(example_data).to_csv(paths["output_dir"] / "blocking_candidate_size_examples.csv", index=False)

    # Distribution CSV
    dist_rows = []
    for label, counts in union_data.items():
        for low, high in DISTRIBUTION_BUCKETS:
            count = ((counts >= low) & (counts < high)).sum()
            dist_rows.append({
                "source": label,
                "bucket": f"{low}-{high}",
                "count": count
            })
    pd.DataFrame(dist_rows).to_csv(paths["output_dir"] / "blocking_candidate_size_distribution.csv", index=False)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
