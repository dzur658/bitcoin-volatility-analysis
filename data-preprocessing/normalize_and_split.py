import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def main():
    master_df = pd.read_csv("data-preprocessing/market-master-df.csv")

    export_dir = "data-preprocessing/normalized-features"

    master_df['open_time'] = pd.to_datetime(master_df['open_time'])
    master_df = master_df.sort_values(by=['open_time', 'symbol']).reset_index(drop=True)

    # Get a unique list of all timestamps and all 21 symbols
    all_times = master_df['open_time'].unique()
    all_symbols = master_df['symbol'].unique()

    # Create a "Perfectly Square" MultiIndex
    # This generates every possible combination of time and symbol
    full_index = pd.MultiIndex.from_product(
        [all_times, all_symbols], 
        names=['open_time', 'symbol']
    )

    # Apply this index to your DataFrame
    # Any coin that didn't exist at a specific timestamp will now get a synthetic row filled with NaNs
    master_df = master_df.set_index(['open_time', 'symbol']).reindex(full_index).reset_index()

    # Re-sort just to be absolutely sure of the chronological layout
    master_df = master_df.sort_values(by=['open_time', 'symbol']).reset_index(drop=True)

    metadata_cols = ['open_time', 'symbol']
    feature_cols = [col for col in master_df.columns if col not in metadata_cols]

    train_mask = master_df['open_time'] < '2025-12-31 00:00:00'
    eval_mask  = (master_df['open_time'] >= '2026-01-01 00:00:00') & (master_df['open_time'] < '2026-03-31 00:00:00')
    test_mask  = (master_df['open_time'] >= '2026-04-01 00:00:00') & (master_df['open_time'] < '2026-07-16 00:00:00')

    validity_df = master_df[['open_time', 'symbol']].copy()
    validity_df[feature_cols] = (~master_df[feature_cols].isna()).astype(int)

    # Initialize and Fit the Scaler ONLY on the Training Split
    scaler = StandardScaler()
    scaler.fit(master_df.loc[train_mask, feature_cols])

    # Transform All Splits using the Training Distribution
    master_df.loc[train_mask, feature_cols] = scaler.transform(master_df.loc[train_mask, feature_cols])
    master_df.loc[eval_mask, feature_cols]  = scaler.transform(master_df.loc[eval_mask, feature_cols])
    master_df.loc[test_mask, feature_cols]  = scaler.transform(master_df.loc[test_mask, feature_cols])

    master_df[feature_cols] = master_df[feature_cols].fillna(0.0)

    # Filter to just the training split
    train_df = master_df[train_mask]
    train_validity = validity_df[train_mask]

    # Pick a feature to test (e.g., 'open')
    feature_to_test = 'open'

    # Extract ONLY the rows where the validity mask is 1 (real data)
    real_data = train_df.loc[train_validity[feature_to_test] == 1, feature_to_test]

    # Check the statistics
    assert np.isclose(real_data.mean(), 0.0), f"Mean is not zero: {real_data.mean()}"
    assert np.isclose(real_data.std(), 1.0), f"Std is not one: {real_data.std()}"

    # make directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # export training dataset and validity mask to parquet
    train_df.to_parquet(os.path.join(export_dir, "train_df.parquet"), index=False)
    train_validity.to_parquet(os.path.join(export_dir, "train_validity.parquet"), index=False)

    # export evaluation dataset and validity mask to parquet
    eval_df = master_df[eval_mask]
    eval_validity = validity_df[eval_mask]

    eval_df.to_parquet(os.path.join(export_dir, "eval_df.parquet"), index=False)
    eval_validity.to_parquet(os.path.join(export_dir, "eval_validity.parquet"), index=False)

    # export test dataset and validity mask to parquet
    test_df = master_df[test_mask]
    test_validity = validity_df[test_mask]

    test_df.to_parquet(os.path.join(export_dir, "test_df.parquet"), index=False)
    test_validity.to_parquet(os.path.join(export_dir, "test_validity.parquet"), index=False)

if __name__ == "__main__":
    main()