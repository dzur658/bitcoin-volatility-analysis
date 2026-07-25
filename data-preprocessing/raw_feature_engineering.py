import pandas as pd
import numpy as np
import os

# prevent division by zero on forward-filled flat bars
EPSILON = 1e-8

def construct_liquidity_features(df):
    """
    Constructs liquidity and volume dynamics, including Amihud Illiquidity.
    """
    # Dollar Volume
    dollar_vol = df['volume'] * df['close']
    
    # Log Volume and Log Dollar Volume
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_dollar_volume'] = np.log(dollar_vol + 1)
    
    # Relative Dollar Volume (21d / 504h)
    mean_dvol_21 = dollar_vol.rolling(window=504).mean()
    df['rel_dvol_21'] = dollar_vol / (mean_dvol_21 + EPSILON)
    
    # Dollar Volume Z-Score (21d / 504h)
    std_dvol_21 = dollar_vol.rolling(window=504).std()
    df['dvol_z_21'] = (dollar_vol - mean_dvol_21) / (std_dvol_21 + EPSILON)
    
    # Amihud Illiquidity (Price impact per dollar traded)
    df['amihud_illiq_1'] = np.abs(df['log_ret_1d']) / (dollar_vol + EPSILON)

    # rolling 21d
    df['amihud_illiq_21'] = df['amihud_illiq_1'].rolling(window=504).mean()
    
    return df

def construct_volatility_features(df):
    """
    Constructs rolling realized volatility and EWMA volatility.
    Fixes the overlapping autocorrelation bias by using discrete 24-hour jumps.
    """
    # 1. REALIZED VOLATILITY (Horizontal Non-Overlapping Std Dev)
    # Collect the last N non-overlapping 24-hour returns
    ret_10d_matrix = pd.concat([df['log_ret_1d'].shift(24 * i) for i in range(10)], axis=1)
    ret_21d_matrix = pd.concat([df['log_ret_1d'].shift(24 * i) for i in range(21)], axis=1)
    ret_63d_matrix = pd.concat([df['log_ret_1d'].shift(24 * i) for i in range(63)], axis=1)
    
    # Calculate Std Dev across the columns (axis=1). No scaling factor needed!
    df['rvol_10'] = ret_10d_matrix.std(axis=1, skipna=False)
    df['rvol_21'] = ret_21d_matrix.std(axis=1, skipna=False)
    df['rvol_63'] = ret_63d_matrix.std(axis=1, skipna=False)
    
    # Regime Ratio
    df['rvol_ratio'] = df['rvol_10'] / (df['rvol_63'] + EPSILON)
    
    # 2. EWMA VOLATILITY (Interleaved Daily Grouping)
    ret_squared = df['log_ret_1d'] ** 2
    
    # Group by the hour of the day to create 24 isolated daily tracks, 
    # apply the halflife, and drop the grouping index to merge it back
    # This effectively locks ewma volatility to timezones ie it is calculated over the same clock-face hour each day
    ewma_10 = ret_squared.groupby(df.index.hour).apply(
        lambda x: x.ewm(halflife=10, min_periods=10).mean()
    ).reset_index(level=0, drop=True)
    
    ewma_20 = ret_squared.groupby(df.index.hour).apply(
        lambda x: x.ewm(halflife=20, min_periods=20).mean()
    ).reset_index(level=0, drop=True)
    
    # Sort index just in case the groupby scrambled the chronological order
    df['ewma_vol_hl10'] = np.sqrt(ewma_10).sort_index()
    df['ewma_vol_hl20'] = np.sqrt(ewma_20).sort_index()
    
    return df

def construct_geometry_features(df):
    """
    Constructs price geometry and wick/shadow features.
    """
    # Log Range
    df['hl_log_range'] = np.log((df['high']) / (df['low']))
    
    # Normalized Body to Range
    body_abs = np.abs(np.log((df['close']) / (df['open'])))
    range_eps = np.log((df['high']) / (df['low']))

    # only add epsilon to this value as the U.S. equities paper describes to preven a possible division by zero error
    df['body_to_range'] = body_abs / (range_eps + EPSILON)
    
    # Upper and Lower Shadows (Wicks)
    max_oc = np.maximum(df['open'], df['close'])
    min_oc = np.minimum(df['open'], df['close'])
    
    df['upper_shadow'] = np.log((df['high']) / (max_oc))
    df['lower_shadow'] = np.log((min_oc) / (df['low']))
    
    return df

def construct_returns_features(df):
    """
    Constructs log returns and horizon structure features.
    """
    # 1d, 5d, 21d, 63d, 126d
    windows = {
        '1d': 24, 
        '5d': 120, 
        '21d': 504, 
        '63d': 1512, 
        '126d': 3024
    }
    
    # Trailing Log Returns
    for name, h in windows.items():
        df[f'log_ret_{name}'] = np.log(df['close'] / df['close'].shift(h))
        
    # Intraday (Intra-bar) and Overnight (Inter-bar)
    df['intraday_ret'] = np.log(df['close'] / (df['open']))
    
    # we drop overnight gap here because it is not relevant for crypto, we have full 24 hour cycles for our coins
    
    # Momentum (126d return - 21d return)
    df['mom_6_1'] = df['log_ret_126d'] - df['log_ret_21d']
    
    return df

def cut_bloat_cols(df):
    bloat_cols = [
        'open_time', 'open_time_ms', 
        'close_time', 'close_time_ms', 
        'interval',
        'quote_volume', 'trades', 
        'taker_buy_base_volume', 'taker_buy_quote_volume'
    ]

    # Safely drop the bloat columns if they exist in the CSV
    cols_to_drop = [col for col in bloat_cols if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean

def pipeline(df):
    """
    Runs the feature engineering pipeline on the input DataFrame.
    """
    df = cut_bloat_cols(df)
    df = construct_returns_features(df)
    df = construct_geometry_features(df)
    df = construct_volatility_features(df)
    df = construct_liquidity_features(df)
    
    return df

def main():
    # ensure the output directory exists
    os.makedirs("data-preprocessing/unnormalized-feature-engineering", exist_ok=True)

    # load the gap filled data from hourly-gap-filling directory
    for file in os.listdir("data-preprocessing/hourly-gap-filling"):
        # ensure gap audit is NOT included
        if file.endswith("_hourly.csv"):
            df_raw = pd.read_csv(os.path.join("data-preprocessing/hourly-gap-filling", file), index_col=0, parse_dates=True)
            
            df_output = pipeline(df_raw)

            # Save the features to a new CSV file
            output_file = os.path.join("data-preprocessing/unnormalized-feature-engineering", f"{file.split('.')[0]}_raw_features.csv")
            df_output.to_csv(output_file, index=True)
            

if __name__ == "__main__":
    main()