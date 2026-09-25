import pandas as pd
from pathlib import Path

from .blocking import (
    create_blocking_keys,
    all_blocking_statistics,
)


def main():

    project_root = Path(__file__).resolve().parents[3]

    path = (
        project_root
        / "processed"
        / "train"
        / "processed_train_source1.tsv"
    )

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5000,
    )

    print(f"Loaded {len(df):,} rows.")

    df = create_blocking_keys(df)

    print("\nBlocking columns:")

    blocking_columns = [
        col
        for col in df.columns
        if col.startswith("block_")
    ]

    for col in blocking_columns:
        print(f"  {col}")

    print("\nBlocking statistics:")

    stats = all_blocking_statistics(df)

    print(
        stats.to_string(index=False)
    )


if __name__ == "__main__":
    main()