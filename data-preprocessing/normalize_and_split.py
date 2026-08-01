import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def main():
    master_df = pd.read_csv("data-preprocessing/market-master-df.csv")

    export_dir = "normalized-features"

    master_df['open_time'] = pd.to_datetime(master_df['open_time'])
    master_df = master_df.sort_values(by=['open_time', 'symbol']).reset_index(drop=True)

    metadata_cols = ['open_time', 'symbol']
    feature_cols = [col for col in master_df.columns if col not in metadata_cols]

    train_mask = master_df['open_time'] < '2025-12-31 00:00:00'
    eval_mask  = (master_df['open_time'] >= '2026-01-01 00:00:00') & (master_df['open_time'] < '2026-03-31 00:00:00')
    test_mask  = (master_df['open_time'] >= '2026-04-01 00:00:00') & (master_df['open_time'] < '2026-07-16 00:00:00')

    master_df["valid_mask"] = (~master_df[feature_cols].isna()).all(axis=1)

    valid_mask = master_df['valid_mask'] == True

    train_valid_mask = train_mask & valid_mask
    eval_valid_mask  = eval_mask & valid_mask
    test_valid_mask  = test_mask & valid_mask

    # Initialize and Fit the Scaler ONLY on the Training Split
    scaler = StandardScaler()
    scaler.fit(master_df.loc[train_valid_mask, feature_cols])

    # Transform All Splits using the Training Distribution
    master_df.loc[train_valid_mask, feature_cols] = scaler.transform(master_df.loc[train_valid_mask, feature_cols])
    master_df.loc[eval_valid_mask, feature_cols]  = scaler.transform(master_df.loc[eval_valid_mask, feature_cols])
    master_df.loc[test_valid_mask, feature_cols]  = scaler.transform(master_df.loc[test_valid_mask, feature_cols])

    normalized_train_df = master_df.loc[train_valid_mask]
    normalized_eval_df  = master_df.loc[eval_valid_mask]
    normalized_test_df  = master_df.loc[test_valid_mask]

    # Pick a feature to test (e.g., 'open')
    feature_to_test = 'log_ret_1d'

    # Extract ONLY the rows where the validity mask is 1 (real data)
    real_data = normalized_train_df[feature_to_test]

    # Check the statistics
    print(f"Mean: {real_data.mean():.6f}")
    print(f"Std Dev: {real_data.std():.6f}")

    # Check the statistics
    assert np.isclose(real_data.mean(), 0.0), f"Mean is not zero: {real_data.mean()}"
    assert np.isclose(real_data.std(), 1.0), f"Std is not one: {real_data.std()}"

    # make directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # export training dataset and validity mask to parquet
    normalized_train_df.to_parquet(os.path.join(export_dir, "train_df.parquet"), index=False)

    normalized_eval_df.to_parquet(os.path.join(export_dir, "eval_df.parquet"), index=False)

    normalized_test_df.to_parquet(os.path.join(export_dir, "test_df.parquet"), index=False)

if __name__ == "__main__":
    main()