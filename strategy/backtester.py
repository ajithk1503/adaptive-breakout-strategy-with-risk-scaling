import pandas as pd
def apply_breakout_strategy(df, atr_multiplier=3):
    df['signal'] = 0
    start_time = pd.to_datetime("09:15").time()
    end_time = pd.to_datetime("10:30").time()

    condition = (
        (df['close'] > df['open'] + atr_multiplier * df['ATR']) &
        (df.index.time >= start_time) &
        (df.index.time <= end_time)
    )
    df.loc[condition, 'signal'] = 1
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def calculate_returns(df, capital=100000):
    df['returns'] = df['close'].pct_change().shift(-1)
    df['strategy_returns'] = df['signal'] * df['returns']
    df['equity'] = (1 + df['strategy_returns'].fillna(0)).cumprod() * capital
    return df
