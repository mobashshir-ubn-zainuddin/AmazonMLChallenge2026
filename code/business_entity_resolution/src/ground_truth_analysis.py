import pandas as pd
from pathlib import Path


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load and parse the training ground truth."""
    df = pd.read_csv(path, sep="\t")

    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    df["matched_ids"] = df["matched_entity_ids"].apply(
        lambda x: [i.strip() for i in x.split(",") if i.strip()]
    )

    df["num_matches"] = df["matched_ids"].apply(len)

    df["num_s2_matches"] = df["matched_ids"].apply(
        lambda ids: sum(i.startswith("S2-") for i in ids)
    )

    df["num_s3_matches"] = df["matched_ids"].apply(
        lambda ids: sum(i.startswith("S3-") for i in ids)
    )

    return df


def main():
    project_root = Path(__file__).resolve().parents[3]

    ground_truth_path = (
        project_root
        / "dataset"
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    print("=" * 60)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 60)

    df = load_ground_truth(ground_truth_path)

    print(f"\nTotal S1 entities in ground truth: {len(df):,}")

    print("\n--- Match Count Distribution ---")
    print(df["num_matches"].value_counts().sort_index())

    print("\n--- S1 Entities by Match Category ---")
    print(f"Zero matches    : {(df['num_matches'] == 0).sum():,}")
    print(f"One match       : {(df['num_matches'] == 1).sum():,}")
    print(f"Multiple matches: {(df['num_matches'] > 1).sum():,}")

    print("\n--- S2 / S3 Match Distribution ---")
    print(f"S1 entities with S2 matches: {(df['num_s2_matches'] > 0).sum():,}")
    print(f"S1 entities with S3 matches: {(df['num_s3_matches'] > 0).sum():,}")

    print("\nTotal S2 matched records:")
    print(df["num_s2_matches"].sum())

    print("Total S3 matched records:")
    print(df["num_s3_matches"].sum())

    print("\n--- Match Count Statistics ---")
    print(f"Average matches per S1: {df['num_matches'].mean():.4f}")
    print(f"Median matches per S1 : {df['num_matches'].median():.0f}")
    print(f"Maximum matches for one S1: {df['num_matches'].max():,}")

    print("\n--- Examples ---")
    print(
        df[
            [
                "source1_entity_id",
                "matched_entity_ids",
                "num_matches",
                "num_s2_matches",
                "num_s3_matches",
            ]
        ].head(10).to_string(index=False)
    )

    output_path = (
        project_root
        / "code"
        / "ground_truth_analysis.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nParsed ground truth saved to:")
    print(output_path)


if __name__ == "__main__":
    main()