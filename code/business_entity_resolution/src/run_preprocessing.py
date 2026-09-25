"""
Script to execute preprocessing on the full dataset.
Processes sources independently to manage memory efficiency.
"""

import pandas as pd
import os
import glob
from . import preprocessing
from pathlib import Path

def run_full_preprocessing():
    # Project root
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Process both train and test datasets
    splits = ["train", "test"]

    for split in splits:
        INPUT_DIR = PROJECT_ROOT / "dataset" / "dataset" / split
        OUTPUT_DIR = PROJECT_ROOT / "processed" / split

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        files = glob.glob(os.path.join(INPUT_DIR, "*.tsv"))

        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} DATA")
        print(f"Input : {INPUT_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Found {len(files)} files")
        print(f"{'=' * 60}")

        for filename in files:
            print(f"Processing {filename}...")

            base_name = os.path.basename(filename)

            # Ground truth is a label file, not a business-record file.
            if base_name == "train_ground_truth.tsv":
                print("Skipping ground truth file during preprocessing.")
                continue

            df = pd.read_csv(filename, sep='\t')
            df['source_file'] = base_name

            processed_df = preprocessing.preprocess_dataframe(df)

            output_path = OUTPUT_DIR / f"processed_{base_name}"

            processed_df.to_csv(
                output_path,
                sep='\t',
                index=False
            )

            print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    run_full_preprocessing()
