import pandas as pd
from binance_api_wrapper import BinanceHistoricalData, Interval

# import target csv
coins = pd.read_csv('data-preprocessing/target_coins.csv')

client = BinanceHistoricalData(
    base_url="https://data-api.binance.vision",
    timeout=30,
    max_retries=5,
    safety_factor=0.9,
)

# iterate through coin USDT pairs and pull historical data
for coin in coins.itertuples():
    # correct date format for API
    month, day, year = coin.Binance_Listing_Date.split('/')

    if len(month) == 1:
        month = f"0{month}"
    if len(day) == 1:
        day = f"0{day}"
        
    api_start_date = f"{year}-{month}-{day}"

    client.fetch_klines_to_csv(
        symbol=coin.Symbol,
        interval=Interval.ONE_HOUR,
        start_time=api_start_date,
        end_time="2026-07-15",
        output_path=f"data-preprocessing/hourly-data-api-out/{coin.Symbol}_hourly.csv"
    )