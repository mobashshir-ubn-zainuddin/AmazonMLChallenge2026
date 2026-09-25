import pandas as pd
import numpy as np
import os
import glob
from typing import List
from pathlib import Path

def load_datasets(dataset_path: str) -> pd.DataFrame:
    """
    Loads the three business source TSV files.
    Ground truth is handled separately.
    """
    all_files = [
        os.path.join(dataset_path, f"train_source{i}.tsv")
        for i in [1, 2, 3]
    ]

    df_list = []

    print(f"Found {len(all_files)} source files.")

    for filename in all_files:
        try:
            df = pd.read_csv(filename, sep='\t', encoding='utf-8')
            df['source_file'] = os.path.basename(filename)
            df_list.append(df)
            print(f"Loaded {filename} with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not df_list:
        raise FileNotFoundError("No source TSV files found.")

    return pd.concat(df_list, ignore_index=True)

def perform_basic_analysis(df: pd.DataFrame):
    """
    Performs basic exploratory data analysis (EDA) on the dataset.
    """
    print("\n--- Dataset Overview ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print("\nColumn Info:")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Unique Values per Column ---")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    if 'country' in df.columns:
        print("\n--- Country Distribution ---")
        print(df['country'].value_counts())

def analyze_text_characteristics(df: pd.DataFrame, column: str):
    """
    Analyzes the characteristics of a text column (e.g., business_name or business_address).
    """
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame.")
        return

    print(f"\n--- Analysis for column: {column} ---")

    # Calculate length of text
    df[f'{column}_len'] = df[column].astype(str).apply(len)

    print(f"Average length: {df[f'{column}_len'].mean():.2f}")
    print(f"Median length: {df[f'{column}_len'].median():.2f}")
    print(f"Max length: {df[f'{column}_len'].max()}")
    print(f"Min length: {df[f'{column}_len'].min()}")

def main():
    # Configuration
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset" / "train"

    try:
        # Step 1: Load Data
        df = load_datasets(DATASET_DIR)

        # Step 2: Basic Analysis
        perform_basic_analysis(df)

        # Step 3: Analyze specific text columns mentioned in the implementation plan
        for col in ['business_name', 'business_address']:
            analyze_text_characteristics(df, col)

        # Save the aggregated dataframe for further use in the pipeline
        output_path = 'code/aggregated_data_analysis.csv'
        df.to_csv(output_path, index=False)
        print(f"\nAggregated data saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
