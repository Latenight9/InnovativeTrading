import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Analysis import (
    load_data,
    calculate_top_correlations,
    run_johansen_tests,
    check_spread_stationarity
)

# 1️⃣ Z-Score berechnen
def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

# 2️⃣ Signale generieren
def generate_signals_from_zscore(zscore: pd.Series, entry=2.0, exit=0.5) -> pd.Series:
    signal = pd.Series(index=zscore.index, data=0)
    signal[zscore > entry] = -1   # Short
    signal[zscore < -entry] = 1   # Long
    signal[(zscore > -exit) & (zscore < exit)] = 0  # Exit
    signal = signal.replace(to_replace=0, method='ffill')  # Position halten
    signal.iloc[-1] = 0  # Letzter Tag immer Flat
    return signal

# 3️⃣ Realistische Trade-Simulation mit Unrealized PnL
def simulate_trades(prices_df: pd.DataFrame, pair: tuple, signals: pd.Series, eigenvector: np.ndarray, initial_capital: float = 10000.0) -> pd.DataFrame:
    stock1, stock2 = pair
    beta = eigenvector[1] / eigenvector[0]

    p1 = prices_df[stock1]
    p2 = prices_df[stock2]

    pos1 = signals.shift(1) * 1
    pos2 = signals.shift(1) * -beta

    returns1 = p1.diff().fillna(0)
    returns2 = p2.diff().fillna(0)

    pnl1 = pos1 * returns1
    pnl2 = pos2 * returns2
    pnl = pnl1 + pnl2

    equity = initial_capital + pnl.cumsum()

    # Unrealized PnL berechnen
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

    trades_df = pd.DataFrame({
        "Stock1_Price": p1,
        "Stock2_Price": p2,
        "Signal": signals,
        "Position_1": pos1,
        "Position_2": pos2,
        "Realized_PnL": pnl,
        "Unrealized_PnL": unrealized_pnl,
        "Equity": equity
    })

    return trades_df

# 4️⃣ Performance auswerten inkl. Unrealized Stats
def evaluate_performance(trades_df: pd.DataFrame):
    equity = trades_df["Equity"]
    returns = equity.pct_change().fillna(0)

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    max_drawdown = (equity / equity.cummax() - 1).min()

    print("\n📊 Performance-Statistiken:")
    print(f"🔹 Finales Kapital:         {equity.iloc[-1]:.2f}")
    print(f"🔹 Gesamtrendite:           {(equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100:.2f}%")
    print(f"🔹 Sharpe Ratio:            {sharpe_ratio:.2f}")
    print(f"🔹 Max. Drawdown:           {max_drawdown * 100:.2f}%")

    max_unrealized = trades_df["Unrealized_PnL"].max()
    min_unrealized = trades_df["Unrealized_PnL"].min()
    print(f"🔹 Höchster offener Gewinn:  {max_unrealized:.2f}")
    print(f"🔹 Höchster offener Verlust: {min_unrealized:.2f}")

# 5️⃣ Plot Equity Curve
def plot_equity(trades_df: pd.DataFrame, pair: tuple):
    trades_df["Equity"].plot(figsize=(12, 5), title=f"Equity Curve – {pair[0]} vs {pair[1]}")
    plt.ylabel("Kapital ($)")
    plt.grid(True)
    plt.show()

# 6️⃣ Plot Unrealized PnL Verlauf
def plot_unrealized(trades_df: pd.DataFrame, pair: tuple):
    trades_df["Unrealized_PnL"].plot(figsize=(12, 4), color="orange", title=f"Unrealized PnL – {pair[0]} vs {pair[1]}")
    plt.ylabel("Offener Gewinn/Verlust ($)")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.show()

# 🔁 Main Flow
if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    prices_df = load_data(start_date, end_date)
    top_pairs = calculate_top_correlations(prices_df)
    johansen_results = run_johansen_tests(top_pairs, prices_df)
    stationary_pairs = check_spread_stationarity(prices_df, johansen_results)

    if not stationary_pairs:
        print("⚠️ Keine stationären Paare gefunden.")
        exit()

    # Nur 1 Paar auswählen für Backtest
    selected_pair = list(stationary_pairs.keys())[0]
    spread = stationary_pairs[selected_pair]["Spread"]
    eigenvector = stationary_pairs[selected_pair]["Eigenvektor"]

    print(f"\n🔁 Backtesting für Paar: {selected_pair}")

    zscore = calculate_zscore(spread)
    signals = generate_signals_from_zscore(zscore)

    trades_df = simulate_trades(
        prices_df=prices_df,
        pair=selected_pair,
        signals=signals,
        eigenvector=eigenvector
    )

    evaluate_performance(trades_df)
    plot_equity(trades_df, selected_pair)
    plot_unrealized(trades_df, selected_pair)
