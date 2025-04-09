import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import START_DATE, END_DATE, INTERVAL, MAX_PAIRS, TOP_N_PAIRS, DATA_MODE
from stock import get_stock_tickers
from crypto import get_crypto_tickers
from Analysis import (
    load_data,
    calculate_top_correlations,
    run_johansen_tests,
    check_spread_stationarity
)

def calculate_zscore(spread, window=20):
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std

def generate_signals_from_zscore(zscore, entry=2.0, exit=0.5):
    signal = pd.Series(index=zscore.index, data=0)
    signal[zscore > entry] = -1
    signal[zscore < -entry] = 1
    signal[(zscore > -exit) & (zscore < exit)] = 0
    signal = signal.replace(to_replace=0, method='ffill')
    signal.iloc[-1] = 0
    return signal

def simulate_trades(prices_df, pair, signals, eigenvector, initial_capital=10000.0):
    stock1, stock2 = pair
    beta = eigenvector[1] / eigenvector[0]

    p1 = prices_df[stock1]
    p2 = prices_df[stock2]

    pos1 = signals * 1
    pos2 = signals * -beta

    returns1 = p1.diff().fillna(0)
    returns2 = p2.diff().fillna(0)

    pnl = pos1 * returns1 + pos2 * returns2
    equity = initial_capital + pnl.cumsum()

    entry_price1 = pd.Series(index=p1.index, dtype='float64')
    entry_price2 = pd.Series(index=p2.index, dtype='float64')
    for i in range(len(signals)):
        if i == 0:
            continue
        if signals.iloc[i] != signals.iloc[i - 1]:
            if signals.iloc[i] != 0:
                entry_price1.iloc[i] = p1.iloc[i]
                entry_price2.iloc[i] = p2.iloc[i]
            else:
                entry_price1.iloc[i] = np.nan
                entry_price2.iloc[i] = np.nan
        else:
            entry_price1.iloc[i] = entry_price1.iloc[i - 1]
            entry_price2.iloc[i] = entry_price2.iloc[i - 1]

    unrealized1 = (p1 - entry_price1) * pos1
    unrealized2 = (p2 - entry_price2) * pos2
    unrealized_pnl = unrealized1 + unrealized2

    return pd.DataFrame({
        "Stock1_Price": p1,
        "Stock2_Price": p2,
        "Signal": signals,
        "Position_1": pos1,
        "Position_2": pos2,
        "Realized_PnL": pnl,
        "Unrealized_PnL": unrealized_pnl,
        "Equity": equity
    })

def evaluate_performance(trades_df):
    equity = trades_df["Equity"]
    returns = equity.pct_change().fillna(0)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    max_drawdown = (equity / equity.cummax() - 1).min()

    print("\n📊 Performance-Statistiken:")
    print(f"🔹 Finales Kapital:         {equity.iloc[-1]:.2f}")
    print(f"🔹 Gesamtrendite:           {(equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100:.2f}%")
    print(f"🔹 Sharpe Ratio:            {sharpe_ratio:.2f}")
    print(f"🔹 Max. Drawdown:           {max_drawdown * 100:.2f}%")
    print(f"🔹 Höchster offener Gewinn:  {trades_df['Unrealized_PnL'].max():.2f}")
    print(f"🔹 Höchster offener Verlust: {trades_df['Unrealized_PnL'].min():.2f}")

def evaluate_portfolio(all_trades):
    combined_equity = sum(df["Equity"] for df in all_trades.values())
    combined_returns = combined_equity.pct_change().fillna(0)

    sharpe_ratio = (combined_returns.mean() / combined_returns.std()) * np.sqrt(252)
    max_drawdown = (combined_equity / combined_equity.cummax() - 1).min()

    print("\n📊 📦 Portfolio Performance:")
    print(f"🔹 Finales Kapital:         {combined_equity.iloc[-1]:.2f}")
    print(f"🔹 Gesamtrendite:           {(combined_equity.iloc[-1] - combined_equity.iloc[0]) / combined_equity.iloc[0] * 100:.2f}%")
    print(f"🔹 Sharpe Ratio:            {sharpe_ratio:.2f}")
    print(f"🔹 Max. Drawdown:           {max_drawdown * 100:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.plot(combined_equity, label="Portfolio Equity")
    plt.title("📦 Gesamt-Equity-Kurve (Portfolio)")
    plt.ylabel("Kapital ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_equity(trades_df, pair):
    trades_df["Equity"].plot(figsize=(12, 5), title=f"Equity Curve – {pair[0]} vs {pair[1]}")
    plt.ylabel("Kapital ($)")
    plt.grid(True)
    plt.show()

# 🔁 Hauptprogramm
if __name__ == "__main__":
    if DATA_MODE == "stocks":
        tickers = get_stock_tickers()
        prices_df = load_data(tickers, START_DATE, END_DATE, interval=INTERVAL)
    else:
        tickers = get_crypto_tickers(n=20)
        prices_df = load_data(tickers, interval=INTERVAL)
    
    
    top_pairs = calculate_top_correlations(prices_df,TOP_N_PAIRS )
    johansen_results = run_johansen_tests(top_pairs, prices_df)
    stationary_pairs = check_spread_stationarity(prices_df, johansen_results)

    if not stationary_pairs:
        print("⚠️ Keine stationären Paare gefunden.")
        exit()

    sorted_pairs = sorted(stationary_pairs.items(), key=lambda x: x[1]["Trace-Statistik"], reverse=True)
    selected_pairs = sorted_pairs[:MAX_PAIRS]

    all_trades = {}
    for pair, data in selected_pairs:
        print(f"\n🔁 Backtesting für Paar: {pair}")
        spread = data["Spread"]
        eigenvector = data["Eigenvektor"]
        zscore = calculate_zscore(spread)
        signals = generate_signals_from_zscore(zscore)
        trades_df = simulate_trades(prices_df, pair, signals, eigenvector)

        evaluate_performance(trades_df)
        plot_equity(trades_df, pair)
        all_trades[pair] = trades_df

    evaluate_portfolio(all_trades)
