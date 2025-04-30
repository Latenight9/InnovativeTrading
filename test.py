from config import START_DATE, END_DATE, INTERVAL, LOOKBACK_PERIOD, INITIAL_CAPITAL
from Analysis import load_data, calculate_rolling_beta, generate_positions_from_zscore 
from backtest import simulate_trades
from chan_example import get_chan_example_tickers

import pandas as pd
import numpy as np

def test_chan_strategy():
    tickers = get_chan_example_tickers()
    prices_df = load_data(tickers, START_DATE, END_DATE, interval=INTERVAL)

    stock1, stock2 = tickers[0], tickers[1]
    p1 = prices_df[stock1]
    p2 = prices_df[stock2]

    print("\n📊 Datenqualität:")
    print(prices_df[[stock1, stock2]].isna().sum())
    print(prices_df[[stock1, stock2]].describe())

    # Rolling Beta
    beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
    print("\n📐 Rolling Beta Qualität:")
    print(f"Valide Werte: {beta_series.count()} / {len(beta_series)}")
    print(beta_series.dropna().head())

    # Spread
    spread = p1 - beta_series * p2
    print("\n🧮 Spread-Statistiken:")
    print(spread.describe())

    # Positionen
    positions = generate_positions_from_zscore(spread, LOOKBACK_PERIOD)
    print("\n📌 Positionen aus Z-Score:")
    print(positions.describe())

    # Trades
    trades_df = simulate_trades(prices_df, (stock1, stock2), positions, beta_series, initial_capital=INITIAL_CAPITAL)
    print("\n💰 Erste 10 Equity-Werte:")
    print(trades_df["Equity"].head(10))

    print("\n📈 Letzte Equity-Werte:")
    print(trades_df["Equity"].tail(10))

    print("\n✅ Test abgeschlossen.")

if __name__ == "__main__":
    test_chan_strategy()
