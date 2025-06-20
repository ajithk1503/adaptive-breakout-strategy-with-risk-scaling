import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve.
    """
    roll_max = equity_curve.cummax()
    drawdown = roll_max - equity_curve
    max_drawdown = drawdown.max()
    return max_drawdown


def calculate_sharpe_ratio(returns, periods_per_year=252 * 375):
    """
    Calculate the annualized Sharpe ratio.
    """
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) using historical method.
    """
    return np.percentile(returns.dropna(), (1 - confidence_level) * 100)


def plot_equity_curve(df, save_path=None, show=True):
    """
    Plot the equity curve from the DataFrame.
    """
    plt.figure(figsize=(10, 5))
    df['equity'].plot(title="Equity Curve", lw=2)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def load_data(filepath, datetime_col="date"):
    """
    Load CSV data, parse datetime, set index.
    """
    df = pd.read_csv(filepath, parse_dates=[datetime_col])
    df = df.rename(columns={datetime_col: "datetime"}).set_index("datetime")
    df = df.sort_index()
    return df
