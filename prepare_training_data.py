import pandas as pd
from data_aggregation import fetch_data, connect_to_mysql_via_ssh
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

# Coins to process
coins = ["ETH", "SUI"]
pd.set_option('display.float_format', '{:.10f}'.format)

try:
    tunnel, connection = connect_to_mysql_via_ssh()

    # Fetch the historical data
    raw_data = fetch_data(connection)

    # Function to calculate technical indicators
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure the data is sorted by timestamp
        df['close_time'] = pd.to_datetime(df['close_time'])
        df = df.sort_values('close_time')
        
        # Calculate EMA 10 and EMA 50
        df['EMA_10'] = EMAIndicator(df['close'], window=10).ema_indicator()
        df['EMA_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        
        # Calculate MACD and MACD Signal
        macd = MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Calculate RSI
        df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
        
        # Calculate VWAP
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['close'],
            low=df['close'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        # Calculate ATR
        df['ATR'] = AverageTrueRange(
            high=df['close'],
            low=df['close'],
            close=df['close'],
            window=14
        ).average_true_range()
        
        # Calculate Bollinger Bands (SMA 20)
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['Bollinger_Middle'] = bb.bollinger_mavg()
        df['Bollinger_Upper'] = bb.bollinger_hband()
        df['Bollinger_Lower'] = bb.bollinger_lband()
        
        # Handle missing values (e.g., at the start of indicators)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    # Prepare data for each coin
    processed_data = {}
    for coin, df in raw_data.items():
        if coin in coins:
            print(f"Processing {coin} data...")
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            processed_data[coin] = calculate_indicators(df)

    # Example output
    for coin, df in processed_data.items():
        print(f"\n{coin} Data with Indicators:")
        print(df.head())

except Exception as e:
        print(f"Connection error: {e}")
    
finally:
    # Clean up resources
    if 'connection' in locals() and connection:
        connection.close()
    if 'tunnel' in locals() and tunnel:
        tunnel.close()