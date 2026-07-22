import pandas as pd
import os

audit_df = pd.DataFrame(columns=["symbol", "count"])

def track_gaps(symbol, gap_count_before_interpolation, gap_count_remaining):
    global audit_df
    audit_df = pd.concat([audit_df, pd.DataFrame({"symbol": [symbol], 
        "prior_count": [gap_count_before_interpolation], 
        "remaining_count": [gap_count_remaining]})], 
        ignore_index=True
    )

def detect_gaps(df):
    price_cols = ['open', 'high', 'low', 'close']
    volume_cols = [
        'volume',
        'quote_volume',
        'trades',
        'taker_buy_base_volume',
        'taker_buy_quote_volume'
    ]

    # Preserve symbol for tracking before reindexing
    symbol = df['symbol'].dropna().iloc[0] if 'symbol' in df.columns else 'UNKNOWN'

    # Set DatetimeIndex and expand to a continuous 1-hour grid
    df.set_index(pd.to_datetime(df['open_time_ms'], unit='ms'), inplace=True)
    df.sort_index(inplace=True)

    df_continuous = df.asfreq('1h')

    # Audit gaps before imputation
    gap_count_before = df_continuous['close'].isna().sum()

    df_continuous['open_time_ms'] = (df_continuous.index.astype('int64'))
    
    # 2. Regenerate the Binance ISO format string (e.g., '2023-03-24T13:00:00Z')
    df_continuous['open_time'] = df_continuous.index.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 3. Regenerate close_time_ms (open + 1 hour - 1 millisecond)
    df_continuous['close_time_ms'] = df_continuous['open_time_ms'] + 3599999

    # 4. Regenerate close_time string (add 59m 59s to index)
    close_dt = df_continuous.index + pd.Timedelta(minutes=59, seconds=59)
    df_continuous['close_time'] = close_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 1. Forward-fill the closing price
    df_continuous['close'] = df_continuous['close'].ffill()

    # 2. Create flat synthetic bars: set open, high, low equal to the carried-forward close
    for col in ['open', 'high', 'low']:
        df_continuous[col] = df_continuous[col].fillna(df_continuous['close'])

    # 3. Set all volume metrics to zero for missing bars
    df_continuous[volume_cols] = df_continuous[volume_cols].fillna(0)

    # Re-fill metadata column if needed
    if 'symbol' in df_continuous.columns:
        df_continuous['symbol'] = df_continuous['symbol'].ffill()
    
    # ensure interval is filled properly
    df_continuous['interval'] = '1h'

    # Check remaining gaps (would only exist if NaNs occurred at the very start of the dataset)
    gap_count_remaining = df_continuous['close'].isna().sum()

    # Log/track the gap audit
    track_gaps(symbol, gap_count_before, gap_count_remaining)

    return df_continuous

def main():
    # make hourly data if it doesn't exist
    if not os.path.exists("data-preprocessing/hourly-gap-filling"):
        os.makedirs("data-preprocessing/hourly-gap-filling")

    # loop through the hourly-data-api-out directory for all csv data on the coins
    for file in os.listdir("data-preprocessing/hourly-data-api-out"):
        if file.endswith(".csv"):
            raw_df = pd.read_csv(os.path.join("data-preprocessing/hourly-data-api-out", file))
            filled_df = detect_gaps(raw_df)
            filled_df.to_csv(os.path.join("data-preprocessing/hourly-gap-filling", file), index=False)
    
    # finally output the audit gap file
    audit_df.to_csv("data-preprocessing/hourly-gap-filling/gap_audit.csv", index=False)

if __name__ == "__main__":
    main()