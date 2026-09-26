"""
Benchmark exact retrieval against the training ground truth.

This is an evaluation-only script.

It:
1. Loads processed training Source 1, Source 2, and Source 3.
2. Builds the exact retrieval index over Source 2 + Source 3.
3. Queries Source 1 records.
4. Measures:
   - pair-level recall
   - S2 recall
   - S3 recall
   - candidate count statistics

It does NOT generate the final submission.
"""

from pathlib import Path
from collections import defaultdict
import time

import pandas as pd

from .exact_index import ExactIndex


ROOT = Path(__file__).resolve().parents[4]

PROCESSED_TRAIN = ROOT / "processed" / "train"
GROUND_TRUTH = (
    ROOT
    / "dataset"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_DIR = ROOT / "code" / "blocking_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


S1_PATH = PROCESSED_TRAIN / "processed_train_source1.tsv"
S2_PATH = PROCESSED_TRAIN / "processed_train_source2.tsv"
S3_PATH = PROCESSED_TRAIN / "processed_train_source3.tsv"


REQUIRED_COLUMNS = [
    "entity_id",
    "country_clean",
    "name_clean",
    "name_no_legal_suffix",
    "address_clean",
]


def load_processed(path: Path) -> pd.DataFrame:
    """Load only columns required for exact retrieval."""

    return pd.read_csv(
        path,
        sep="\t",
        usecols=REQUIRED_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )


def load_ground_truth() -> dict:
    """
    Load ground truth as:

        source1_id -> set(matched_entity_ids)
    """

    gt = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in gt.itertuples(index=False):
        matched = (
            set(row.matched_entity_ids.split(","))
            if row.matched_entity_ids
            else set()
        )

        truth[row.source1_entity_id] = matched

    return truth


def percentile(values, q):
    """Calculate a percentile without requiring scipy."""

    return float(pd.Series(values).quantile(q))


def main():

    start_time = time.time()

    print("=" * 70)
    print("EXACT RETRIEVAL BENCHMARK")
    print("=" * 70)

    print("\nLoading processed training data...")

    s1 = load_processed(S1_PATH)
    s2 = load_processed(S2_PATH)
    s3 = load_processed(S3_PATH)

    print(f"S1 rows: {len(s1):,}")
    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    expected_s1 = 2_206_821
    expected_s2 = 5_034_616
    expected_s3 = 5_285_603

    assert len(s1) == expected_s1
    assert len(s2) == expected_s2
    assert len(s3) == expected_s3

    print("\nLoading ground truth...")

    truth = load_ground_truth()

    total_true_pairs = sum(
        len(matches)
        for matches in truth.values()
    )

    total_true_s2 = sum(
        sum(entity_id.startswith("S2-") for entity_id in matches)
        for matches in truth.values()
    )

    total_true_s3 = sum(
        sum(entity_id.startswith("S3-") for entity_id in matches)
        for matches in truth.values()
    )

    print(f"Total true pairs: {total_true_pairs:,}")
    print(f"True S2 pairs:    {total_true_s2:,}")
    print(f"True S3 pairs:    {total_true_s3:,}")

    # --------------------------------------------------------------
    # Build exact index
    # --------------------------------------------------------------

    print("\nBuilding exact retrieval index...")

    index = ExactIndex()

    index_start = time.time()

    index.add_dataframe(s2)
    index.add_dataframe(s3)

    index_time = time.time() - index_start

    print(f"Index construction time: {index_time:.2f} seconds")

    print("\nIndex statistics:")

    for key, value in index.statistics().items():
        print(f"  {key}: {value:,}")

    # --------------------------------------------------------------
    # Query S1
    # --------------------------------------------------------------

    print("\nQuerying Source 1...")

    query_start = time.time()

    total_candidates = 0
    candidate_counts = []

    retrieved_pairs = set()

    for row in s1.itertuples(index=False):

        candidates = index.retrieve_record(row)

        candidate_count = len(candidates)

        total_candidates += candidate_count
        candidate_counts.append(candidate_count)

        for entity_id in candidates:
            retrieved_pairs.add(
                (row.entity_id, entity_id)
            )

    query_time = time.time() - query_start

    print(f"Query time: {query_time:.2f} seconds")

    # --------------------------------------------------------------
    # Recall
    # --------------------------------------------------------------

    true_pairs = set()

    for source1_id, matched_ids in truth.items():

        for entity_id in matched_ids:
            true_pairs.add(
                (source1_id, entity_id)
            )

    recovered_pairs = true_pairs.intersection(
        retrieved_pairs
    )

    missed_pairs = true_pairs - retrieved_pairs

    pair_recall = (
        len(recovered_pairs) / len(true_pairs)
        if true_pairs
        else 0.0
    )

    recovered_s2 = sum(
        entity_id.startswith("S2-")
        for _, entity_id in recovered_pairs
    )

    recovered_s3 = sum(
        entity_id.startswith("S3-")
        for _, entity_id in recovered_pairs
    )

    s2_recall = (
        recovered_s2 / total_true_s2
        if total_true_s2
        else 0.0
    )

    s3_recall = (
        recovered_s3 / total_true_s3
        if total_true_s3
        else 0.0
    )

    # --------------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------------

    candidate_series = pd.Series(
        candidate_counts,
        dtype="int64",
    )

    candidate_stats = {
        "total_s1": len(s1),
        "total_candidate_pairs": total_candidates,
        "mean_candidates": candidate_series.mean(),
        "median_candidates": candidate_series.median(),
        "p75_candidates": candidate_series.quantile(0.75),
        "p90_candidates": candidate_series.quantile(0.90),
        "p95_candidates": candidate_series.quantile(0.95),
        "p99_candidates": candidate_series.quantile(0.99),
        "p999_candidates": candidate_series.quantile(0.999),
        "max_candidates": candidate_series.max(),
        "zero_candidate_s1": int(
            (candidate_series == 0).sum()
        ),
    }

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXACT RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nRecovered true pairs: "
        f"{len(recovered_pairs):,} / {len(true_pairs):,}"
    )

    print(f"Missed true pairs:    {len(missed_pairs):,}")

    print(f"\nOverall pair recall: {pair_recall:.4%}")
    print(f"S2 pair recall:      {s2_recall:.4%}")
    print(f"S3 pair recall:      {s3_recall:.4%}")

    print("\nCandidate statistics:")

    for key, value in candidate_stats.items():

        if isinstance(value, float):
            print(f"  {key}: {value:,.4f}")
        else:
            print(f"  {key}: {value:,}")

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "retrieval_method": "exact",
                "true_pairs": len(true_pairs),
                "recovered_pairs": len(recovered_pairs),
                "missed_pairs": len(missed_pairs),
                "pair_recall": pair_recall,
                "s2_true_pairs": total_true_s2,
                "s2_recovered_pairs": recovered_s2,
                "s2_recall": s2_recall,
                "s3_true_pairs": total_true_s3,
                "s3_recovered_pairs": recovered_s3,
                "s3_recall": s3_recall,
                **candidate_stats,
                "index_time_seconds": index_time,
                "query_time_seconds": query_time,
                "total_time_seconds": time.time() - start_time,
            }
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "exact_retrieval_benchmark.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved benchmark to:\n"
        f"{output_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()