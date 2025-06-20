# adaptive-breakout-strategy-with-risk-scaling

## Intraday Volatility-Based Breakout Strategy with Risk-Aware Position Sizing

This project implements and backtests a simple yet effective intraday breakout strategy using 1-minute OHLC data and Average True Range (ATR) as a volatility filter. Built in Python, the pipeline logs all performance metrics with MLflow, and supports modular development with indicator and utility modules.

### Strategy Overview

1. Type: Intraday Breakout Strategy

2. Instrument: Nifty Futures or any high-volume instrument with 1-min data

3. Entry Time: Between 09:15 AM and 10:30 AM

4. Logic: Buy when close > open + ATR * N

5. Risk Control: Signal must execute on next candle, fixed capital model
