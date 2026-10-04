import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", limit=1000, start_str=None, end_str=None):
    """
    Fetches historical kline (candlestick) data from Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    
    # Convert string dates to milliseconds timestamp
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000) if start_str else None
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000) if end_str else None

    all_klines = []
    current_start = start_ts

    print(f"Fetching {symbol} {interval} data from Binance...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": current_start,
            "endTime": end_ts
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_klines.extend(data)
        print(f"Fetched {len(all_klines)} candles...")
        
        # Update current_start to the last candle's timestamp + 1 millisecond
        current_start = data[-1][0] + 1
        
        if end_ts and current_start >= end_ts:
            break
            
        # Respect API rate limits
        time.sleep(0.5)

    # Convert to DataFrame
    columns = [
        "timestamp", "Open", "High", "Low", "Close", "Volume", 
        "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    
    # Clean up and format types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Keep only what we need and convert to float (tb_base_av is Taker Buy Base Asset Volume)
    df = df[["Open", "High", "Low", "Close", "Volume", "tb_base_av"]].astype(float)
    return df

def calculate_vwap_features(df):
    """
    Applies our deterministic quantitative filters.
    Uses Rolling-14 VWAP (the proven working system the model was originally trained on).
    """
    print("Calculating Rolling-14 VWAP & Momentum Features...")
    df = df.copy()
    
    # 1. Rolling-14 VWAP (proven working system)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['VWAP'] = df['VP'].rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()
    
    # 2. Distance to VWAP
    df['Dist_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # 3. Momentum (Rate of Change over 5 periods)
    df['Momentum_5'] = df['Close'].pct_change(periods=5)
    
    # 4. Volume Surge
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Surge'] = df['Volume'] / df['Volume_MA_20']
    
    # 5. Cumulative Volume Delta (CVD)
    # Taker Buy = Aggressive Longs. Taker Sell = Aggressive Shorts.
    df['Taker_Sell_Volume'] = df['Volume'] - df['tb_base_av']
    df['Net_Volume'] = df['tb_base_av'] - df['Taker_Sell_Volume']
    df['CVD_14'] = df['Net_Volume'].rolling(window=14).sum()
    
    df = df.drop(columns=['Typical_Price', 'VP', 'Volume_MA_20', 'tb_base_av', 'Taker_Sell_Volume', 'Net_Volume'])
    df = df.dropna()
    return df

if __name__ == "__main__":
    # Fetch 1 Year of 1-minute data for Cloud Training
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df_raw = fetch_binance_klines(
        symbol="BTCUSDT", 
        interval="1m", 
        start_str=start_date.strftime("%Y-%m-%d"), 
        end_str=end_date.strftime("%Y-%m-%d")
    )
    
    df_features = calculate_vwap_features(df_raw)
    
    # Save to CSV in a format Qlib's dump script expects
    # Qlib requires date, symbol, and features
    df_features['symbol'] = 'BTCUSDT'
    df_features = df_features.reset_index()
    df_features.rename(columns={'timestamp': 'date'}, inplace=True)
    
    csv_name = "btc_1m_qlib_ready.csv"
    df_features.to_csv(csv_name, index=False)
    
    print(f"\nSuccess! Saved {len(df_features)} rows of high-frequency data to {csv_name}.")
    print("Data is now ready to be ingested into Qlib's .bin format.")
