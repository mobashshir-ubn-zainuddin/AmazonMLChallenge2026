"""
Evaluate blocking recall on the training ground truth.

Important:
This script evaluates whether true S1 -> S2/S3 pairs share at least
one blocking key.

It does NOT generate the final candidate_pairs.tsv.

The purpose is to determine the recall ceiling of the blocking stage
before we build the full candidate-generation pipeline.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .blocking import (
    BLOCKING_STRATEGIES,
    create_blocking_keys,
    pair_survives_blocking,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 100_000

REQUIRED_SOURCE_COLUMNS = [
    "entity_id",
    "country_clean",
    "name_clean",
    "name_no_legal_suffix",
    "name_sorted_tokens",
    "name_tokens",
    "address_clean",
    "address_numbers",
    "address_tokens",
]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def get_paths():
    """Return all project paths."""

    project_root = Path(__file__).resolve().parents[3]

    paths = {
        "project_root": project_root,

        "train_dir": (
            project_root
            / "dataset"
            / "dataset"
            / "train"
        ),

        "processed_train_dir": (
            project_root
            / "processed"
            / "train"
        ),

        "ground_truth": (
            project_root
            / "dataset"
            / "dataset"
            / "train"
            / "train_ground_truth.tsv"
        ),

        "output_dir": (
            project_root
            / "code"
            / "blocking_results"
        ),
    }

    return paths


# ---------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------

def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load ground truth and convert the comma-separated match list
    into one row per true pair.

    Output columns:

        source1_entity_id
        matched_entity_id
    """

    print(f"Loading ground truth: {path}")

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
        .astype(str)
    )

    # Explode comma-separated IDs.
    gt["matched_entity_id"] = (
        gt["matched_entity_ids"]
        .str.split(",")
    )

    gt = gt[
        [
            "source1_entity_id",
            "matched_entity_id",
        ]
    ].explode(
        "matched_entity_id",
        ignore_index=True,
    )

    gt["matched_entity_id"] = (
        gt["matched_entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove S1 rows with no match.
    gt = gt[
        gt["matched_entity_id"] != ""
    ].copy()

    print(
        f"True matched pairs: {len(gt):,}"
    )

    return gt


# ---------------------------------------------------------------------
# Load processed source data
# ---------------------------------------------------------------------

def load_processed_source(
    path: Path,
) -> pd.DataFrame:
    """
    Load only the columns required for blocking.

    This function is intended for one source at a time.
    """

    print(f"Loading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    missing = [
        col
        for col in REQUIRED_SOURCE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    df = df[REQUIRED_SOURCE_COLUMNS].copy()

    return df


# ---------------------------------------------------------------------
# Prepare blocking data
# ---------------------------------------------------------------------

def prepare_source(
    path: Path,
) -> pd.DataFrame:
    """
    Load a processed source and create blocking keys.
    """

    df = load_processed_source(path)

    print(
        f"Loaded {len(df):,} records."
    )

    df = create_blocking_keys(df)

    return df


# ---------------------------------------------------------------------
# Evaluate one source
# ---------------------------------------------------------------------

def evaluate_source(
    source1_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Evaluate blocking recall for one candidate source.

    source_name is either S2 or S3.

    Only true ground-truth pairs belonging to this source
    are evaluated.
    """

    print("\n" + "=" * 70)
    print(
        f"Evaluating blocking recall against {source_name}"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Restrict ground truth to this source.
    # -------------------------------------------------------------

    gt = ground_truth[
        ground_truth["matched_entity_id"]
        .str.startswith(source_name + "-")
    ].copy()

    print(
        f"Ground-truth {source_name} pairs: "
        f"{len(gt):,}"
    )

    if gt.empty:
        return pd.DataFrame()

    # -------------------------------------------------------------
    # Create lookup tables.
    #
    # Only true matched candidate records are required.
    # -------------------------------------------------------------

    source1_lookup = (
        source1_df
        .set_index("entity_id")
    )

    candidate_lookup = (
        candidate_df
        .set_index("entity_id")
    )

    # -------------------------------------------------------------
    # Evaluate every true pair.
    # -------------------------------------------------------------

    counts = {
        strategy: 0
        for strategy in BLOCKING_STRATEGIES
    }

    any_block_count = 0

    missing_s1 = 0
    missing_candidate = 0

    total_pairs = len(gt)

    # Iterate through ground-truth pairs.
    #
    # This is deliberately simple and transparent.
    # We will optimize further if profiling shows that it is
    # necessary.
    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.matched_entity_id

        if s1_id not in source1_lookup.index:
            missing_s1 += 1
            continue

        if candidate_id not in candidate_lookup.index:
            missing_candidate += 1
            continue

        s1_row = source1_lookup.loc[s1_id]
        candidate_row = candidate_lookup.loc[candidate_id]

        result = pair_survives_blocking(
            s1_row,
            candidate_row,
        )

        survived_any = False

        for strategy, survived in result.items():

            if survived:
                counts[strategy] += 1
                survived_any = True

        if survived_any:
            any_block_count += 1

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    results = []

    for strategy in BLOCKING_STRATEGIES:

        recovered = counts[strategy]

        recall = (
            recovered / total_pairs
            if total_pairs > 0
            else 0.0
        )

        results.append(
            {
                "source": source_name,
                "strategy": strategy,
                "true_pairs": total_pairs,
                "recovered_pairs": recovered,
                "blocking_recall": recall,
            }
        )

    union_recall = (
        any_block_count / total_pairs
        if total_pairs > 0
        else 0.0
    )

    results.append(
        {
            "source": source_name,
            "strategy": "UNION_ALL",
            "true_pairs": total_pairs,
            "recovered_pairs": any_block_count,
            "blocking_recall": union_recall,
        }
    )

    print("\n--- Blocking Recall ---")

    for result in results:

        print(
            f"{result['strategy']:25s} "
            f"{result['recovered_pairs']:>10,} / "
            f"{result['true_pairs']:>10,} "
            f"= "
            f"{result['blocking_recall'] * 100:8.3f}%"
        )

    if missing_s1:
        print(
            f"\nWARNING: Missing S1 records: "
            f"{missing_s1:,}"
        )

    if missing_candidate:
        print(
            f"WARNING: Missing {source_name} records: "
            f"{missing_candidate:,}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("BLOCKING RECALL ANALYSIS")
    print("=" * 70)

    paths = get_paths()

    paths["output_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # 1. Load ground truth
    # -------------------------------------------------------------

    ground_truth = load_ground_truth(
        paths["ground_truth"]
    )

    # -------------------------------------------------------------
    # 2. Load Source 1
    # -------------------------------------------------------------

    source1_path = (
        paths["processed_train_dir"]
        / "processed_train_source1.tsv"
    )

    source1 = prepare_source(
        source1_path
    )

    print(
        f"\nSource 1 records: "
        f"{len(source1):,}"
    )

    # -------------------------------------------------------------
    # 3. Evaluate S2
    # -------------------------------------------------------------

    source2_path = (
        paths["processed_train_dir"]
        / "processed_train_source2.tsv"
    )

    source2 = prepare_source(
        source2_path
    )

    results_s2 = evaluate_source(
        source1,
        source2,
        ground_truth,
        "S2",
    )

    # Free Source 2 before loading Source 3.
    del source2

    # -------------------------------------------------------------
    # 4. Evaluate S3
    # -------------------------------------------------------------

    source3_path = (
        paths["processed_train_dir"]
        / "processed_train_source3.tsv"
    )

    source3 = prepare_source(
        source3_path
    )

    results_s3 = evaluate_source(
        source1,
        source3,
        ground_truth,
        "S3",
    )

    del source3
    del source1

    # -------------------------------------------------------------
    # 5. Combine results
    # -------------------------------------------------------------

    results = pd.concat(
        [
            results_s2,
            results_s3,
        ],
        ignore_index=True,
    )

    output_path = (
        paths["output_dir"]
        / "blocking_recall_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()