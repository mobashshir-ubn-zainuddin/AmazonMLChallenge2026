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
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from business_entity_resolution.src.blocking import create_blocking_keys
from business_entity_resolution.src.retrieval.exact_index import ExactIndex
from business_entity_resolution.src.retrieval.token_index import TokenIndex


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "dataset" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

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
        for item in value.split(";")
        if item.strip()
    }


def percentile(values: List[int], q: float) -> float:
    """Compute percentile without requiring NumPy."""
    if not values:
        return 0.0

    values_sorted = sorted(values)

    if len(values_sorted) == 1:
        return float(values_sorted[0])

    position = (len(values_sorted) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(values_sorted[lower])

    fraction = position - lower

    return (
        values_sorted[lower]
        + fraction * (values_sorted[upper] - values_sorted[lower])
    )


def candidate_statistics(candidate_counts: List[int]) -> Dict[str, float]:
    """Calculate candidate-count statistics."""
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

    return {
        "mean": statistics.fmean(candidate_counts),
        "median": statistics.median(candidate_counts),
        "p90": percentile(candidate_counts, 0.90),
        "p95": percentile(candidate_counts, 0.95),
        "p99": percentile(candidate_counts, 0.99),
        "max": max(candidate_counts),
        "zero": sum(x == 0 for x in candidate_counts),
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

def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth into:

        source1_entity_id -> set(matched_entity_ids)

    This is approximately 2.2M dictionary entries and is required
    for S1-by-S1 evaluation.
    """
    print("Loading ground truth...")

    gt = pd.read_csv(
        path,
        sep="\t",
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

    ground_truth: Dict[str, Set[str]] = {}

    for row in gt.itertuples(index=False):
        source1_id = normalize_id(row.source1_entity_id)
        matched = parse_matched_ids(row.matched_entity_ids)

        ground_truth[source1_id] = matched

    print(f"Ground-truth S1 records: {len(ground_truth):,}")

    return ground_truth


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

        self.candidate_counts: List[int] = []

        self.s1_recall_values: List[float] = []

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

        self.s1_recall_values.append(recall)

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
        if not self.s1_recall_values:
            return 0.0

        return statistics.fmean(self.s1_recall_values)

    def candidate_stats(self) -> Dict[str, float]:
        return candidate_statistics(self.candidate_counts)


def evaluate_retrieval(
    s1: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
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

        truth = ground_truth.get(
            source1_id,
            set(),
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

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
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
        len(matches)
        for matches in ground_truth.values()
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