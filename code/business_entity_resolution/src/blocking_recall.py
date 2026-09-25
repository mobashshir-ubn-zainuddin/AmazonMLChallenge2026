"""
Vectorized blocking-recall evaluation for Amazon ML Challenge 2026.

Evaluates whether known ground-truth S1 -> S2/S3 pairs share
each blocking key.

This script does NOT generate candidate_pairs.tsv.

It measures:

1. Recall of each blocking strategy
2. Recall of the UNION of all strategies
3. Number of true pairs recovered

The evaluation is performed in chunks to avoid constructing
the complete multi-million-row pair table in memory.
"""

from pathlib import Path

import pandas as pd

from .blocking import (
    BLOCKING_STRATEGIES,
    _longest_token,
    _first_address_number,
)


# ============================================================
# Configuration
# ============================================================

GT_CHUNK_SIZE = 100_000

STRATEGIES = BLOCKING_STRATEGIES


# ============================================================
# Paths
# ============================================================

def get_paths():

    project_root = Path(__file__).resolve().parents[3]

    return {
        "project_root": project_root,

        "ground_truth": (
            project_root
            / "dataset"
            / "dataset"
            / "train"
            / "train_ground_truth.tsv"
        ),

        "processed_train": (
            project_root
            / "processed"
            / "train"
        ),

        "output_dir": (
            project_root
            / "code"
            / "blocking_results"
        ),
    }


# ============================================================
# Mapping between strategy and blocking column
# ============================================================

def strategy_column(strategy):

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

    return mapping[strategy]


# ============================================================
# Read only one blocking key from a processed source
# ============================================================
def longest_token_series(series):
    """
    Reproduce the exact longest-token logic used by blocking.py.
    """
    return series.apply(_longest_token)

def load_key_column(path, strategy):
    """
    Load only the columns required to construct one blocking key.

    Blocking keys are generated here from the already-preprocessed
    columns instead of expecting block_* columns to exist on disk.
    """

    required_columns = {
        "name_exact": [
            "entity_id",
            "country_clean",
            "name_clean",
        ],
        "name_no_suffix_exact": [
            "entity_id",
            "country_clean",
            "name_no_legal_suffix",
        ],
        "name_sorted_exact": [
            "entity_id",
            "country_clean",
            "name_sorted_tokens",
        ],
        "name_prefix4": [
            "entity_id",
            "country_clean",
            "name_no_legal_suffix",
        ],
        "name_token": [
            "entity_id",
            "country_clean",
            "name_tokens",
        ],
        "address_exact": [
            "entity_id",
            "country_clean",
            "address_clean",
        ],
        "address_numbers": [
            "entity_id",
            "country_clean",
            "address_numbers",
        ],
        "address_token": [
            "entity_id",
            "country_clean",
            "address_tokens",
        ],
        "name_address_exact": [
            "entity_id",
            "country_clean",
            "name_clean",
            "address_clean",
        ],
        "name_no_suffix_address": [
            "entity_id",
            "country_clean",
            "name_no_legal_suffix",
            "address_clean",
        ],
        "name_no_suffix_address_numbers": [
            "entity_id",
            "country_clean",
            "name_no_legal_suffix",
            "address_numbers",
        ],
        "name_prefix4_address_number": [
            "entity_id",
            "country_clean",
            "name_no_legal_suffix",
            "address_numbers",
        ],
    }

    if strategy not in required_columns:
        raise ValueError(f"Unknown blocking strategy: {strategy}")

    columns = required_columns[strategy]

    print(f"Loading {path.name} for strategy [{strategy}]...")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype=str,
        keep_default_na=False,
    )

    country = df["country_clean"]

    if strategy == "name_exact":
        name = df["name_clean"]

        key = (
            country + "|" + name
        ).where(
            country.ne("") & name.ne(""),
            ""
        )

    elif strategy == "name_no_suffix_exact":
        name_no_suffix = df["name_no_legal_suffix"]

        key = (
            country + "|" + name_no_suffix
        ).where(
            country.ne("") & name_no_suffix.ne(""),
            ""
        )

    elif strategy == "name_sorted_exact":
        name_sorted = df["name_sorted_tokens"]

        key = (
            country + "|" + name_sorted
        ).where(
            country.ne("") & name_sorted.ne(""),
            ""
        )

    elif strategy == "name_prefix4":
        name_prefix4 = df["name_no_legal_suffix"].str[:4]

        key = (
            country + "|" + name_prefix4
        ).where(
            country.ne("") & name_prefix4.ne(""),
            ""
        )

    elif strategy == "name_token":
        longest_name = longest_token_series(
            df["name_tokens"]
        )

        key = (
            country + "|" + longest_name
        ).where(
            country.ne("") & longest_name.ne(""),
            ""
        )

    elif strategy == "address_exact":
        address = df["address_clean"]

        key = (
            country + "|" + address
        ).where(
            country.ne("") & address.ne(""),
            ""
        )

    elif strategy == "address_numbers":
        address_numbers = df["address_numbers"]

        key = (
            country + "|" + address_numbers
        ).where(
            country.ne("") & address_numbers.ne(""),
            ""
        )

    elif strategy == "address_token":
        longest_address = longest_token_series(
            df["address_tokens"]
        )

        key = (
            country + "|" + longest_address
        ).where(
            country.ne("") & longest_address.ne(""),
            ""
        )

    elif strategy == "name_address_exact":
        name = df["name_clean"]
        address = df["address_clean"]

        key = (
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

    elif strategy == "name_no_suffix_address":
        name_no_suffix = df["name_no_legal_suffix"]
        address = df["address_clean"]

        key = (
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

    elif strategy == "name_no_suffix_address_numbers":
        name_no_suffix = df["name_no_legal_suffix"]
        address_numbers = df["address_numbers"]

        key = (
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

    elif strategy == "name_prefix4_address_number":
        name_prefix4 = df["name_no_legal_suffix"].str[:4]
        first_address_number = df["address_numbers"].apply(
            _first_address_number
        )

        key = (
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

    else:
        raise ValueError(f"Unhandled strategy: {strategy}")

    result = pd.DataFrame({
        "entity_id": df["entity_id"],
        "block_key": key,
    })

    return result

# ============================================================
# Load and expand ground truth
# ============================================================

def load_ground_truth_pairs(path):

    print(
        f"\nLoading ground truth: {path}"
    )

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

    gt = gt[
        gt["matched_entity_id"] != ""
    ].copy()

    gt["source"] = (
        gt["matched_entity_id"]
        .str[:2]
    )

    print(
        f"Total true matched pairs: "
        f"{len(gt):,}"
    )

    print(
        f"S2 true pairs: "
        f"{(gt['source'] == 'S2').sum():,}"
    )

    print(
        f"S3 true pairs: "
        f"{(gt['source'] == 'S3').sum():,}"
    )

    return gt


# ============================================================
# Evaluate one blocking strategy
# ============================================================

def evaluate_strategy(
    strategy,
    gt,
    source1_path,
    source2_path,
    source3_path,
):

    column = strategy_column(strategy)

    print("\n" + "=" * 75)
    print(
        f"STRATEGY: {strategy}"
    )
    print(
        f"COLUMN : {column}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load only the relevant key from Source 1.
    # --------------------------------------------------------

    s1 = load_key_column(
        source1_path,
        strategy,
    )

    s1 = s1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "block_key": "s1_key",
        }
    )

    # --------------------------------------------------------
    # Evaluate S2 and S3 separately.
    # --------------------------------------------------------

    results = []

    for source_name, source_path in [
        ("S2", source2_path),
        ("S3", source3_path),
    ]:

        print(
            f"\nEvaluating {source_name}..."
        )

        # Only load the relevant blocking key.
        candidate = load_key_column(
            source_path,
            strategy,
        )

        candidate = candidate.rename(
            columns={
                "entity_id": "matched_entity_id",
                "block_key": "candidate_key",
            }
        )

        # ----------------------------------------------------
        # Restrict ground truth to this source.
        # ----------------------------------------------------

        source_gt = gt[
            gt["source"] == source_name
        ][
            [
                "source1_entity_id",
                "matched_entity_id",
            ]
        ].copy().reset_index(drop=True)

        total_pairs = len(source_gt)

        # ----------------------------------------------------
        # Merge S1 blocking key.
        # ----------------------------------------------------

        source_gt = source_gt.merge(
            s1,
            on="source1_entity_id",
            how="left",
            sort=False,
        )

        # ----------------------------------------------------
        # Merge candidate blocking key.
        # ----------------------------------------------------

        source_gt = source_gt.merge(
            candidate,
            on="matched_entity_id",
            how="left",
            sort=False,
        )

        # ----------------------------------------------------
        # A true pair survives if:
        #
        #   S1 key == candidate key
        #
        # and neither key is empty.
        # ----------------------------------------------------

        valid = (
            source_gt["s1_key"].ne("")
            &
            source_gt["candidate_key"].ne("")
            &
            source_gt["s1_key"].eq(
                source_gt["candidate_key"]
            )
        )

        recovered = int(valid.sum())

        recall = (
            recovered / total_pairs
            if total_pairs
            else 0.0
        )

        print(
            f"{source_name}: "
            f"{recovered:,} / "
            f"{total_pairs:,} "
            f"= {recall * 100:.4f}%"
        )

        results.append(
            {
                "strategy": strategy,
                "source": source_name,
                "true_pairs": total_pairs,
                "recovered_pairs": recovered,
                "blocking_recall": recall,
            }
        )

        del candidate
        del source_gt

    del s1

    return results


# ============================================================
# Evaluate UNION of all blocking strategies
# ============================================================

def evaluate_union(
    gt,
    source1_path,
    source2_path,
    source3_path,
):

    print("\n" + "=" * 75)
    print("UNION OF ALL BLOCKING STRATEGIES")
    print("=" * 75)

    # --------------------------------------------------------
    # For each strategy, evaluate whether a true pair survives.
    #
    # Store only boolean recovery flags per strategy/source.
    # --------------------------------------------------------

    union_results = []

    for source_name, source_path in [
        ("S2", source2_path),
        ("S3", source3_path),
    ]:

        source_gt = gt[
            gt["source"] == source_name
        ][
            [
                "source1_entity_id",
                "matched_entity_id",
            ]
        ].copy().reset_index(drop=True)

        source_gt["survived_any"] = False

        for strategy in STRATEGIES:

            column = strategy_column(strategy)

            print(
                f"\n{source_name} | "
                f"Checking {strategy}"
            )

            # ------------------------------------------------
            # Load S1 key.
            # ------------------------------------------------

            s1 = load_key_column(
                source1_path,
                strategy,
            )

            s1 = s1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "block_key": "s1_key",
        }
    )

            # ------------------------------------------------
            # Load candidate key.
            # ------------------------------------------------

            candidate = load_key_column(
                source_path,
                strategy,
            )

            candidate = candidate.rename(
                columns={
                    "entity_id": "matched_entity_id",
                    "block_key": "candidate_key",
                }
            )

            # ------------------------------------------------
            # Merge S1 key.
            # ------------------------------------------------

            temp = source_gt[
                [
                    "source1_entity_id",
                    "matched_entity_id",
                ]
            ].merge(
                s1,
                on="source1_entity_id",
                how="left",
                sort=False,
            )

            # ------------------------------------------------
            # Merge candidate key.
            # ------------------------------------------------

            temp = temp.merge(
                candidate,
                on="matched_entity_id",
                how="left",
                sort=False,
            )

            survived = (
                temp["s1_key"].ne("")
                &
                temp["candidate_key"].ne("")
                &
                temp["s1_key"].eq(
                    temp["candidate_key"]
                )
            )

            source_gt.loc[
                survived.index,
                "survived_any"
            ] |= survived.to_numpy()

            recovered = int(
                source_gt["survived_any"].sum()
            )

            print(
                f"Current UNION recall: "
                f"{recovered:,} / "
                f"{len(source_gt):,} "
                f"= "
                f"{recovered / len(source_gt) * 100:.4f}%"
            )

            del s1
            del candidate
            del temp

        total_pairs = len(source_gt)

        recovered = int(
            source_gt["survived_any"].sum()
        )

        recall = (
            recovered / total_pairs
            if total_pairs
            else 0.0
        )

        union_results.append(
            {
                "strategy": "UNION_ALL",
                "source": source_name,
                "true_pairs": total_pairs,
                "recovered_pairs": recovered,
                "blocking_recall": recall,
            }
        )

    return union_results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("BLOCKING RECALL ANALYSIS")
    print("=" * 75)

    paths = get_paths()

    paths["output_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    source1_path = (
        paths["processed_train"]
        / "processed_train_source1.tsv"
    )

    source2_path = (
        paths["processed_train"]
        / "processed_train_source2.tsv"
    )

    source3_path = (
        paths["processed_train"]
        / "processed_train_source3.tsv"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    gt = load_ground_truth_pairs(
        paths["ground_truth"]
    )
    
    # --------------------------------------------------------
    # Evaluate individual strategies
    # --------------------------------------------------------

    all_results = []

    for strategy in STRATEGIES:

        results = evaluate_strategy(
            strategy,
            gt,
            source1_path,
            source2_path,
            source3_path,
        )

        all_results.extend(results)

    # --------------------------------------------------------
    # Evaluate UNION
    # --------------------------------------------------------

    union_results = evaluate_union(
        gt,
        source1_path,
        source2_path,
        source3_path,
    )

    all_results.extend(
        union_results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        paths["output_dir"]
        / "blocking_recall_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL BLOCKING RECALL RESULTS")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()