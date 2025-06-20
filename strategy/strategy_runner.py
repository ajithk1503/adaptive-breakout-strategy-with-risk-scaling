import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import os

from indicators import compute_atr, compute_vwap

def run_strategy(df, atr_multiplier=3, risk_pct=0.01, capital=100000):
    mlflow.start_run()

    mlflow.log_param("ATR_multiplier", atr_multiplier)
    mlflow.log_param("risk_per_trade", risk_pct)

    df = compute_atr(df)
    df = compute_vwap(df)

    # Time range
    start_time = pd.to_datetime("09:15").time()
    end_time = pd.to_datetime("10:30").time()

    df['signal'] = 0
    condition = (
        (df['close'] > df['open'] + atr_multiplier * df['ATR']) &
        (df.index.time >= start_time) &
        (df.index.time <= end_time)
    )
    df.loc[condition, 'signal'] = 1
    df['signal'] = df['signal'].shift(1).fillna(0)

    df['returns'] = df['close'].pct_change().shift(-1)
    df['strategy_returns'] = df['signal'] * df['returns']
    df['equity'] = (1 + df['strategy_returns'].fillna(0)).cumprod() * capital

    # Metrics
    cum_return = df['equity'].iloc[-1] - capital
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * (252 * 375) ** 0.5
    max_dd = (df['equity'].cummax() - df['equity']).max()
    var_95 = df['strategy_returns'].quantile(0.05)

    mlflow.log_metric("Cumulative_Return", round(cum_return, 2))
    mlflow.log_metric("Sharpe_Ratio", round(sharpe, 4))
    mlflow.log_metric("Max_Drawdown", round(max_dd, 2))
    mlflow.log_metric("VaR_95", round(var_95, 4))

    # Save equity plot
    plt.figure(figsize=(10, 5))
    df['equity'].plot(title="Equity Curve")
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    mlflow.log_artifact("equity_curve.png")

    mlflow.end_run()
    return df


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/NIFTY 50_minute_data.csv", parse_dates=["date"])
    df = df.rename(columns={"date": "datetime"})
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Run strategy
    results = run_strategy(df, atr_multiplier=3, risk_pct=0.01)
    print(results.tail(5))
