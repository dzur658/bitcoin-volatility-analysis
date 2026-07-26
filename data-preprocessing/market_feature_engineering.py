import os
import pandas as pd
import numpy as np

EPSILON = 1e-8

def construct_market_relative_features(df):
    df["mkt_log_ret_1"] = df.groupby("open_time")["log_ret_1d"].transform("mean")

    df["ret_vs_mkt_1"] = df["log_ret_1d"] - df["mkt_log_ret_1"]

    df["mkt_rvol_21"] = df.groupby("open_time")["rvol_21"].transform("mean")

    df["rel_vol"] = df["rvol_21"] / (df["mkt_rvol_21"] + EPSILON)

    # Market relative features should not be added until 2017-12-12
    # see the first candle timestamp notebook for reasoning on why this date in particular was chosen
    threshold_date = "2017-12-12"
    market_cols = ["mkt_log_ret_1", "ret_vs_mkt_1", "mkt_rvol_21", "rel_vol"]

    df.loc[df["open_time"] < threshold_date, market_cols] = np.nan

    return df

def main():
    dataframes = []

    for file in os.listdir("data-preprocessing/unnormalized-feature-engineering"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join("data-preprocessing/unnormalized-feature-engineering", file))
            dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    master_df["open_time"] = pd.to_datetime(master_df["open_time"])
    master_df = master_df.sort_values(by=["open_time", "symbol"]).reset_index(drop=True)

    market_master_df = construct_market_relative_features(master_df)

    pd.DataFrame.to_csv(market_master_df, "data-preprocessing/market-master-df.csv", index=False)

if __name__ == "__main__":
    main()