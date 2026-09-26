"""
Controlled benchmark for token retrieval.

Runs on a configurable subset of Source 1 records before scaling
to the full training dataset.
"""

from pathlib import Path
import time

import pandas as pd

from .exact_index import ExactIndex
from .token_index import TokenIndex


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


S1_COLUMNS = [
    "entity_id",
    "country_clean",
    "name_clean",
    "name_no_legal_suffix",
    "address_clean",
    "name_tokens",
    "address_tokens",
]

INDEX_COLUMNS = [
    "entity_id",
    "country_clean",
    "name_clean",
    "name_no_legal_suffix",
    "address_clean",
    "name_tokens",
    "address_tokens",
]


# ------------------------------------------------------------
# EXPERIMENT SETTINGS
# ------------------------------------------------------------

S1_SAMPLE_SIZE = 50_000

TOKEN_MAX_DF = 10_000
TOKEN_TOP_K = 5
TOKEN_MAX_CANDIDATES = 200


def load_data(path: Path, columns):

    return pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype=str,
        keep_default_na=False,
    )


def load_ground_truth():

    gt = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in gt.itertuples(index=False):

        truth[row.source1_entity_id] = (
            set(row.matched_entity_ids.split(","))
            if row.matched_entity_ids
            else set()
        )

    return truth


def evaluate(
    s1,
    truth,
    exact_index,
    token_index,
):

    true_pairs = set()

    for row in s1.itertuples(index=False):

        for entity_id in truth.get(
            row.entity_id,
            set(),
        ):

            true_pairs.add(
                (row.entity_id, entity_id)
            )

    print(
        f"\nTrue pairs in sample: "
        f"{len(true_pairs):,}"
    )

    results = []

    # --------------------------------------------------------
    # TOKEN ONLY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKEN RETRIEVAL ONLY")
    print("=" * 70)

    start = time.time()

    retrieved_pairs = set()
    candidate_counts = []

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1,
    ):

        candidates = token_index.retrieve_record(
            row,
            max_df=TOKEN_MAX_DF,
            top_k_tokens=TOKEN_TOP_K,
            max_candidates=TOKEN_MAX_CANDIDATES,
        )

        candidate_counts.append(len(candidates))

        for entity_id in candidates:

            retrieved_pairs.add(
                (row.entity_id, entity_id)
            )

        if i % 10_000 == 0:

            elapsed = time.time() - start

            print(
                f"Processed {i:,}/{len(s1):,} "
                f"({i / len(s1):.1%}) "
                f"| elapsed {elapsed:.1f}s"
            )

    elapsed = time.time() - start

    recovered = true_pairs.intersection(
        retrieved_pairs
    )

    series = pd.Series(
        candidate_counts,
        dtype="int64",
    )

    token_result = {
        "retrieval_method": "token",
        "sample_s1": len(s1),
        "true_pairs": len(true_pairs),
        "recovered_pairs": len(recovered),
        "missed_pairs": len(true_pairs - retrieved_pairs),
        "pair_recall": (
            len(recovered) / len(true_pairs)
            if true_pairs
            else 0.0
        ),
        "total_candidate_pairs": int(series.sum()),
        "mean_candidates": series.mean(),
        "median_candidates": series.median(),
        "p90_candidates": series.quantile(0.90),
        "p95_candidates": series.quantile(0.95),
        "p99_candidates": series.quantile(0.99),
        "max_candidates": series.max(),
        "zero_candidate_s1": int(
            (series == 0).sum()
        ),
        "elapsed_seconds": elapsed,
    }

    results.append(token_result)

    for key, value in token_result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # EXACT + TOKEN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXACT + TOKEN RETRIEVAL")
    print("=" * 70)

    start = time.time()

    retrieved_pairs = set()
    candidate_counts = []

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1,
    ):

        exact_candidates = (
            exact_index.retrieve_record(row)
        )

        token_candidates = (
            token_index.retrieve_record(
                row,
                max_df=TOKEN_MAX_DF,
                top_k_tokens=TOKEN_TOP_K,
                max_candidates=TOKEN_MAX_CANDIDATES,
            )
        )

        candidates = set(exact_candidates)

        candidates.update(
            token_candidates
        )

        candidate_counts.append(
            len(candidates)
        )

        for entity_id in candidates:

            retrieved_pairs.add(
                (row.entity_id, entity_id)
            )

        if i % 10_000 == 0:

            elapsed = time.time() - start

            print(
                f"Processed {i:,}/{len(s1):,} "
                f"({i / len(s1):.1%}) "
                f"| elapsed {elapsed:.1f}s"
            )

    elapsed = time.time() - start

    recovered = true_pairs.intersection(
        retrieved_pairs
    )

    series = pd.Series(
        candidate_counts,
        dtype="int64",
    )

    union_result = {
        "retrieval_method": "exact+token",
        "sample_s1": len(s1),
        "true_pairs": len(true_pairs),
        "recovered_pairs": len(recovered),
        "missed_pairs": len(true_pairs - retrieved_pairs),
        "pair_recall": (
            len(recovered) / len(true_pairs)
            if true_pairs
            else 0.0
        ),
        "total_candidate_pairs": int(series.sum()),
        "mean_candidates": series.mean(),
        "median_candidates": series.median(),
        "p90_candidates": series.quantile(0.90),
        "p95_candidates": series.quantile(0.95),
        "p99_candidates": series.quantile(0.99),
        "max_candidates": series.max(),
        "zero_candidate_s1": int(
            (series == 0).sum()
        ),
        "elapsed_seconds": elapsed,
    }

    results.append(union_result)

    for key, value in union_result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR
        / "token_retrieval_benchmark_50k.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved benchmark to:\n"
        f"{output_path}"
    )


def main():

    print("=" * 70)
    print("CONTROLLED TOKEN RETRIEVAL BENCHMARK")
    print("=" * 70)

    print(
        f"\nS1 sample size: "
        f"{S1_SAMPLE_SIZE:,}"
    )

    print(
        f"Token max_df: "
        f"{TOKEN_MAX_DF:,}"
    )

    print(
        f"Token top_k: "
        f"{TOKEN_TOP_K}"
    )

    print(
        f"Token max candidates: "
        f"{TOKEN_MAX_CANDIDATES}"
    )

    print("\nLoading data...")

    s1 = load_data(
        S1_PATH,
        S1_COLUMNS,
    )

    s2 = load_data(
        S2_PATH,
        INDEX_COLUMNS,
    )

    s3 = load_data(
        S3_PATH,
        INDEX_COLUMNS,
    )

    # Deterministic sample
    s1 = s1.head(
        S1_SAMPLE_SIZE
    ).copy()

    print(f"S1 used: {len(s1):,}")
    print(f"S2 index records: {len(s2):,}")
    print(f"S3 index records: {len(s3):,}")

    print("\nLoading ground truth...")

    truth = load_ground_truth()

    print(
        f"Ground-truth S1 records: "
        f"{len(truth):,}"
    )

    # --------------------------------------------------------
    # BUILD EXACT INDEX
    # --------------------------------------------------------

    print("\nBuilding exact index...")

    start = time.time()

    exact_index = ExactIndex()

    exact_index.add_dataframe(s2)
    exact_index.add_dataframe(s3)

    print(
        f"Exact index time: "
        f"{time.time() - start:.2f}s"
    )

    # --------------------------------------------------------
    # BUILD TOKEN INDEX
    # --------------------------------------------------------

    print("\nBuilding token index...")

    start = time.time()

    token_index = TokenIndex()

    token_index.add_dataframe(s2)
    token_index.add_dataframe(s3)

    print(
        f"Token index time: "
        f"{time.time() - start:.2f}s"
    )

    print("\nToken index statistics:")

    for key, value in token_index.statistics().items():

        print(
            f"  {key}: {value:,}"
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    evaluate(
        s1=s1,
        truth=truth,
        exact_index=exact_index,
        token_index=token_index,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()