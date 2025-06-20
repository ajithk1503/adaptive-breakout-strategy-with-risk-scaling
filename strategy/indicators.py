import pandas as pd
import numpy as np

def compute_atr(df, period=14):
    """
    Average True Range (ATR)
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    df['ATR'] = atr
    return df

def compute_vwap(df):
    """
    Volume Weighted Average Price (VWAP)
    """
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df.drop(['cum_vol', 'cum_vol_price'], axis=1, inplace=True)
    return df

def compute_ema(df, column='close', period=20):
    """
    Exponential Moving Average (EMA)
    """
    df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
    return df

def compute_rsi(df, column='close', period=14):
    """
    Relative Strength Index (RSI)
    """
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df
