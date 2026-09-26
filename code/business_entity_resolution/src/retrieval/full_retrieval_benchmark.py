"""
Full-data retrieval benchmark.

Evaluates:
    1. Exact retrieval
    2. Token retrieval
    3. Exact + Token retrieval

over the complete training dataset.

Also evaluates:
    - candidate-count statistics
    - per-S1 recall
    - old blocking vs Exact + Token overlap

Designed to avoid storing all retrieved pairs globally.

Run:
    $env:PYTHONPATH="$PWD\code"
    python -m business_entity_resolution.src.retrieval.full_retrieval_benchmark
"""

from __future__ import annotations

import csv
import time
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from business_entity_resolution.src.blocking import create_blocking_keys
from business_entity_resolution.src.retrieval.exact_index import ExactIndex
from business_entity_resolution.src.retrieval.token_index import TokenIndex


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROCESSED_TRAIN_DIR = PROJECT_ROOT / "processed" / "train"
GROUND_TRUTH_DIR = PROJECT_ROOT / "dataset" / "dataset" / "train"

S1_PATH = PROCESSED_TRAIN_DIR / "processed_train_source1.tsv"
S2_PATH = PROCESSED_TRAIN_DIR / "processed_train_source2.tsv"
S3_PATH = PROCESSED_TRAIN_DIR / "processed_train_source3.tsv"
GT_PATH = GROUND_TRUTH_DIR / "train_ground_truth.tsv"

RESULT_DIR = PROJECT_ROOT / "code" / "blocking_results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Current validated token settings.
TOKEN_MAX_DF = 10_000
TOKEN_TOP_K = 5
TOKEN_MAX_CANDIDATES = 200

# Progress display.
PROGRESS_EVERY = 10_000


# ============================================================
# Utility functions
# ============================================================

def normalize_id(value) -> str:
    """Convert an entity ID to a consistent string."""
    if pd.isna(value):
        return ""
    return str(value)


def parse_matched_ids(value) -> Set[str]:
    """
    Parse the matched_entity_ids field.

    The ground-truth file uses a delimiter-separated list of IDs.
    Empty / missing values represent zero matches.
    """
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    # Ground truth normally uses ';'.
    # Keep a small amount of robustness for whitespace.
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def percentile(values, q: float) -> float:
    """Calculate an exact percentile using a compact NumPy array."""
    if len(values) == 0:
        return 0.0

    values_array = np.asarray(values, dtype=np.int64)
    return float(np.quantile(values_array, q))


def candidate_statistics(candidate_counts: array) -> Dict[str, float]:
    """
    Calculate candidate-count statistics without retaining a Python list
    of millions of Python integers.

    The resulting statistics use the same percentile definitions as the
    original implementation.
    """
    if not candidate_counts:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "zero": 0,
        }

    values_array = np.asarray(candidate_counts, dtype=np.int64)

    return {
        "mean": float(values_array.mean()),
        "median": float(np.quantile(values_array, 0.50)),
        "p90": float(np.quantile(values_array, 0.90)),
        "p95": float(np.quantile(values_array, 0.95)),
        "p99": float(np.quantile(values_array, 0.99)),
        "max": int(values_array.max()),
        "zero": int(np.count_nonzero(values_array == 0)),
    }


def print_candidate_statistics(stats: Dict[str, float]) -> None:
    """Print candidate statistics."""
    print(f"mean_candidates: {stats['mean']}")
    print(f"median_candidates: {stats['median']}")
    print(f"p90_candidates: {stats['p90']}")
    print(f"p95_candidates: {stats['p95']}")
    print(f"p99_candidates: {stats['p99']}")
    print(f"max_candidates: {stats['max']}")
    print(f"zero_candidate_s1: {stats['zero']}")


# ============================================================
# Ground-truth loading
# ============================================================

def load_ground_truth(path: Path) -> pd.Series:
    """
    Load ground truth as a memory-efficient indexed pandas Series:

        source1_entity_id -> matched_entity_ids string

    The previous implementation materialized approximately 2.2M Python
    dictionary entries and a Python set for every S1 record. That creates
    substantial object overhead. Keeping the delimiter-separated match IDs
    as strings preserves the same information while parsing only the
    current S1 record during evaluation.
    """
    print("Loading ground truth...")

    gt = pd.read_csv(
        path,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(gt.columns)

    if missing:
        raise ValueError(
            f"Ground-truth file is missing columns: {sorted(missing)}"
        )

    gt["source1_entity_id"] = gt["source1_entity_id"].map(normalize_id)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    # The challenge has one ground-truth row per Source 1 entity.
    # Keep the same ID-based lookup semantics as the original dict.
    ground_truth = gt.set_index("source1_entity_id")["matched_entity_ids"]

    print(f"Ground-truth S1 records: {len(ground_truth):,}")

    return ground_truth


def ground_truth_matches(ground_truth: pd.Series, source1_id: str) -> Set[str]:
    """
    Parse only one S1 record's matched IDs at evaluation time.

    This preserves the original set-based intersection semantics without
    storing 2.2M Python sets simultaneously.
    """
    value = ground_truth.get(source1_id, "")
    return parse_matched_ids(value)


# ============================================================
# Index construction
# ============================================================

def build_exact_index(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> ExactIndex:
    """Build exact retrieval index over S2 + S3."""

    print()
    print("Building exact index...")

    start = time.perf_counter()

    index = ExactIndex()

    index.add_dataframe(s2)
    index.add_dataframe(s3)

    elapsed = time.perf_counter() - start

    print(f"Exact index time: {elapsed:.2f}s")

    return index


def build_token_index(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> TokenIndex:
    """Build token retrieval index over S2 + S3."""

    print()
    print("Building token index...")

    start = time.perf_counter()

    index = TokenIndex()

    index.add_dataframe(s2)
    index.add_dataframe(s3)

    index.finalize_index()

    elapsed = time.perf_counter() - start

    print(f"Token index time: {elapsed:.2f}s")

    print()
    print("Token index statistics:")

    stats = index.statistics()

    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    return index


# ============================================================
# Retrieval
# ============================================================

def retrieve_exact(
    exact_index: ExactIndex,
    row,
) -> Set[str]:
    """Retrieve candidates using exact retrieval."""
    candidates = exact_index.retrieve_record(
        row,
    )

    return set(candidates)


def retrieve_token(
    token_index: TokenIndex,
    row,
) -> Set[str]:
    """Retrieve candidates using token retrieval."""
    candidates = token_index.retrieve_record(
        row,
        max_df=TOKEN_MAX_DF,
        top_k_tokens=TOKEN_TOP_K,
        max_candidates=TOKEN_MAX_CANDIDATES,
    )

    return set(candidates)


def retrieve_exact_token(
    exact_index: ExactIndex,
    token_index: TokenIndex,
    row,
) -> Set[str]:
    """Union exact and token candidates."""
    exact_candidates = retrieve_exact(
        exact_index,
        row,
    )

    token_candidates = retrieve_token(
        token_index,
        row,
    )

    return exact_candidates | token_candidates


# ============================================================
# Per-S1 evaluation
# ============================================================

class EvaluationAccumulator:
    """
    Accumulates full-dataset retrieval statistics without storing
    all retrieved pairs.
    """

    def __init__(self) -> None:
        self.total_true_pairs = 0
        self.total_recovered_pairs = 0

        # Candidate counts are bounded by the retrieval configuration and
        # are therefore stored as compact unsigned integers.
        self.candidate_counts = array("I")

        # Keep only the sum/count needed for the macro S1 recall.
        self.s1_recall_sum = 0.0
        self.s1_recall_count = 0

        self.zero_recall_s1 = 0
        self.lt_50_recall_s1 = 0
        self.lt_80_recall_s1 = 0
        self.lt_90_recall_s1 = 0
        self.lt_100_recall_s1 = 0
        self.full_recall_s1 = 0

    def update(
        self,
        truth: Set[str],
        candidates: Set[str],
    ) -> None:
        true_count = len(truth)

        recovered_count = len(truth & candidates)

        self.total_true_pairs += true_count
        self.total_recovered_pairs += recovered_count

        self.candidate_counts.append(len(candidates))

        # S1-level recall.
        #
        # For zero-match S1 records, retrieval recall is treated
        # as 1.0 because there are no true pairs to miss.
        if true_count == 0:
            recall = 1.0
        else:
            recall = recovered_count / true_count

        self.s1_recall_sum += recall
        self.s1_recall_count += 1

        if recall == 0.0:
            self.zero_recall_s1 += 1

        if recall < 0.5:
            self.lt_50_recall_s1 += 1

        if recall < 0.8:
            self.lt_80_recall_s1 += 1

        if recall < 0.9:
            self.lt_90_recall_s1 += 1

        if recall < 1.0:
            self.lt_100_recall_s1 += 1

        if recall == 1.0:
            self.full_recall_s1 += 1

    @property
    def missed_pairs(self) -> int:
        return self.total_true_pairs - self.total_recovered_pairs

    @property
    def pair_recall(self) -> float:
        if self.total_true_pairs == 0:
            return 0.0

        return self.total_recovered_pairs / self.total_true_pairs

    @property
    def macro_s1_recall(self) -> float:
        if self.s1_recall_count == 0:
            return 0.0

        return self.s1_recall_sum / self.s1_recall_count

    def candidate_stats(self) -> Dict[str, float]:
        return candidate_statistics(self.candidate_counts)


def evaluate_retrieval(
    s1: pd.DataFrame,
    ground_truth: pd.Series,
    retrieve_function,
    method_name: str,
) -> EvaluationAccumulator:
    """Evaluate one retrieval method over all S1 records."""

    print()
    print("=" * 60)
    print(method_name)
    print("=" * 60)

    accumulator = EvaluationAccumulator()

    start = time.perf_counter()

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1,
    ):
        source1_id = normalize_id(row.entity_id)

        truth = ground_truth_matches(
            ground_truth,
            source1_id,
        )

        candidates = retrieve_function(row)

        accumulator.update(
            truth,
            candidates,
        )

        if i % PROGRESS_EVERY == 0 or i == len(s1):
            elapsed = time.perf_counter() - start

            percentage = (
                100.0 * i / len(s1)
                if len(s1)
                else 100.0
            )

            print(
                f"Processed {i:,}/{len(s1):,} "
                f"({percentage:.1f}%) | "
                f"elapsed {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - start

    print()
    print(f"retrieval_method: {method_name}")
    print(f"s1_records: {len(s1):,}")
    print(f"true_pairs: {accumulator.total_true_pairs:,}")
    print(f"recovered_pairs: {accumulator.total_recovered_pairs:,}")
    print(f"missed_pairs: {accumulator.missed_pairs:,}")
    print(f"pair_recall: {accumulator.pair_recall:.10f}")
    print(f"macro_s1_recall: {accumulator.macro_s1_recall:.10f}")

    print()
    print("Candidate statistics:")
    print_candidate_statistics(
        accumulator.candidate_stats()
    )

    print()
    print("Per-S1 recall:")
    print(f"0.0 recall S1s: {accumulator.zero_recall_s1:,}")
    print(f"< 0.5 recall S1s: {accumulator.lt_50_recall_s1:,}")
    print(f"< 0.8 recall S1s: {accumulator.lt_80_recall_s1:,}")
    print(f"< 0.9 recall S1s: {accumulator.lt_90_recall_s1:,}")
    print(f"< 1.0 recall S1s: {accumulator.lt_100_recall_s1:,}")
    print(f"1.0 recall S1s: {accumulator.full_recall_s1:,}")

    print()
    print(f"elapsed_seconds: {elapsed:.2f}")

    return accumulator


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 60)
    print("FULL RETRIEVAL BENCHMARK")
    print("=" * 60)

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading data...")

    # Load only columns consumed by the exact and token indexes plus S1 ID.
    # This avoids carrying unrelated processed columns through the full run.
    required_data_columns = [
        "entity_id",
        "country_clean",
        "name_clean",
        "name_no_legal_suffix",
        "address_clean",
        "name_tokens",
        "address_tokens",
    ]

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        usecols=required_data_columns,
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        usecols=required_data_columns,
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        usecols=required_data_columns,
        dtype=str,
        keep_default_na=False,
    )

    print(f"S1 records: {len(s1):,}")
    print(f"S2 records: {len(s2):,}")
    print(f"S3 records: {len(s3):,}")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth(GT_PATH)

    total_true_pairs = sum(
        len(parse_matched_ids(value))
        for value in ground_truth.values
    )

    print(
        f"Ground-truth pairs: {total_true_pairs:,}"
    )

    # --------------------------------------------------------
    # Validate S1 ID column
    # --------------------------------------------------------

    if "entity_id" not in s1.columns:
        raise ValueError(
            "Source 1 must contain 'entity_id'."
        )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    exact_index = build_exact_index(
        s2,
        s3,
    )

    token_index = build_token_index(
        s2,
        s3,
    )
    # S2 and S3 DataFrames are no longer needed.
    # Both retrieval indexes have already been constructed.
    import gc

    del s2
    del s3
    gc.collect()

    print("Released S2/S3 DataFrames from memory.")
    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    exact_result = evaluate_retrieval(
        s1=s1,
        ground_truth=ground_truth,
        retrieve_function=lambda row: retrieve_exact(
            exact_index,
            row,
        ),
        method_name="EXACT",
    )

    # --------------------------------------------------------
    # TOKEN
    # --------------------------------------------------------

    token_result = evaluate_retrieval(
        s1=s1,
        ground_truth=ground_truth,
        retrieve_function=lambda row: retrieve_token(
            token_index,
            row,
        ),
        method_name="TOKEN",
    )

    # --------------------------------------------------------
    # EXACT + TOKEN
    # --------------------------------------------------------

    exact_token_result = evaluate_retrieval(
        s1=s1,
        ground_truth=ground_truth,
        retrieve_function=lambda row: retrieve_exact_token(
            exact_index,
            token_index,
            row,
        ),
        method_name="EXACT + TOKEN",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_elapsed = time.perf_counter() - total_start

    print()
    print("=" * 60)
    print("FULL RETRIEVAL SUMMARY")
    print("=" * 60)

    print()
    print(
        f"{'Method':<20}"
        f"{'Recovered':>15}"
        f"{'Missed':>15}"
        f"{'Recall':>15}"
        f"{'Mean Cand.':>15}"
    )

    print("-" * 80)

    for name, result in [
        ("Exact", exact_result),
        ("Token", token_result),
        ("Exact + Token", exact_token_result),
    ]:
        stats = result.candidate_stats()

        print(
            f"{name:<20}"
            f"{result.total_recovered_pairs:>15,}"
            f"{result.missed_pairs:>15,}"
            f"{result.pair_recall:>14.6%}"
            f"{stats['mean']:>15.2f}"
        )

    print()
    print(
        f"Total benchmark time: "
        f"{total_elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        RESULT_DIR
        / "full_retrieval_benchmark_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "method",
                "s1_records",
                "true_pairs",
                "recovered_pairs",
                "missed_pairs",
                "pair_recall",
                "macro_s1_recall",
                "mean_candidates",
                "median_candidates",
                "p90_candidates",
                "p95_candidates",
                "p99_candidates",
                "max_candidates",
                "zero_candidate_s1",
                "zero_recall_s1",
                "lt_50_recall_s1",
                "lt_80_recall_s1",
                "lt_90_recall_s1",
                "lt_100_recall_s1",
                "full_recall_s1",
            ]
        )

        for name, result in [
            ("exact", exact_result),
            ("token", token_result),
            ("exact+token", exact_token_result),
        ]:

            stats = result.candidate_stats()

            writer.writerow(
                [
                    name,
                    len(s1),
                    result.total_true_pairs,
                    result.total_recovered_pairs,
                    result.missed_pairs,
                    result.pair_recall,
                    result.macro_s1_recall,
                    stats["mean"],
                    stats["median"],
                    stats["p90"],
                    stats["p95"],
                    stats["p99"],
                    stats["max"],
                    stats["zero"],
                    result.zero_recall_s1,
                    result.lt_50_recall_s1,
                    result.lt_80_recall_s1,
                    result.lt_90_recall_s1,
                    result.lt_100_recall_s1,
                    result.full_recall_s1,
                ]
            )

    print()
    print(
        f"Saved summary to:\n{summary_path}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()