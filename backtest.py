import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import START_DATE, END_DATE, INTERVAL, MAX_PAIRS, TOP_N_PAIRS, DATA_MODE, INITIAL_CAPITAL, ZSCORE_ENTRY, ZSCORE_EXIT, LOOKBACK_PERIOD
from stock import get_stock_tickers
from crypto import get_crypto_tickers
from chan_example import get_chan_example_tickers
from Analysis import (
    load_data,
    calculate_top_correlations,
    run_johansen_tests,
    check_spread_stationarity,
    calculate_rolling_beta,
    generate_positions_from_zscore
)

def calculate_zscore(spread, window=30):
    rolling_mean = spread.shift(1).rolling(window=window).mean() #lookahead bias vermeiden
    rolling_std = spread.shift(1).rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.fillna(0)
    return zscore

def generate_signals_from_zscore(zscore, entry=2.0, exit=0.5):
    signal = pd.Series(index=zscore.index, data=0)
    signal[zscore > entry] = -1
    signal[zscore < -entry] = 1
    signal[(zscore > -exit) & (zscore < exit)] = 0
    signal = signal.replace(0, np.nan).ffill()
    signal.iloc[0] = 0
    signal.iloc[-1] = 0
    return signal

def simulate_trades(prices_df, pair, positions, beta_series, initial_capital=10000.0):
    stock1, stock2 = pair
    p1 = prices_df[stock1]
    p2 = prices_df[stock2]

    pos1_shares = positions
    pos2_shares = -beta_series * positions

    pos1_dollar = pos1_shares * p1
    pos2_dollar = pos2_shares * p2

    ret1 = p1.pct_change().fillna(0)
    ret2 = p2.pct_change().fillna(0)

    pnl = pos1_dollar.shift(1) * ret1 + pos2_dollar.shift(1) * ret2
    pnl = pnl.fillna(0)

    # Equity in $ wie bisher
    equity = initial_capital + pnl.cumsum()

    # Chan-style: Return bezogen auf investiertes Kapital
    gross_market_value = pos1_dollar.shift(1).abs() + pos2_dollar.shift(1).abs()
    ret = pnl / gross_market_value
    ret = ret.fillna(0)

    cumulative_return = (1 + ret).cumprod() - 1

    # Einstandspreise zur Berechnung offener PnL
    entry_price1 = p1.copy()
    entry_price2 = p2.copy()
    for i in range(1, len(positions)):
        if positions.iloc[i] != positions.iloc[i - 1]:
            if positions.iloc[i] != 0:
                entry_price1.iloc[i] = p1.iloc[i]
                entry_price2.iloc[i] = p2.iloc[i]
            else:
                entry_price1.iloc[i] = np.nan
                entry_price2.iloc[i] = np.nan
        else:
            entry_price1.iloc[i] = entry_price1.iloc[i - 1]
            entry_price2.iloc[i] = entry_price2.iloc[i - 1]

    unrealized1 = (p1 - entry_price1) * pos1_shares
    unrealized2 = (p2 - entry_price2) * pos2_shares
    unrealized_pnl = (unrealized1 + unrealized2).fillna(0)

    return pd.DataFrame({
        "Stock1_Price": p1,
        "Stock2_Price": p2,
        "Position_1": pos1_shares,
        "Position_2": pos2_shares,
        "Position_$1": pos1_dollar,
        "Position_$2": pos2_dollar,
        "Realized_PnL": pnl,
        "Unrealized_PnL": unrealized_pnl,
        "Equity": equity,
        "Return": ret,
        "Cumulative_Return": cumulative_return
    })



def evaluate_performance(trades_df):
    returns = trades_df["Return"]
    cumulative_return = trades_df["Cumulative_Return"]
    unrealized = trades_df["Unrealized_PnL"]

    mean_daily_return = returns.mean()
    std_daily_return = returns.std()

    apr = (1 + mean_daily_return) ** 252 - 1
    sharpe = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0
    max_drawdown = (cumulative_return - cumulative_return.cummax()).min()

    print("\n📊 Performance-Statistiken (Chan-kompatibel):")
    print(f"🔹 APR:                    {apr:.2%}")
    print(f"🔹 Sharpe Ratio:           {sharpe:.2f}")
    print(f"🔹 Max. Drawdown:          {max_drawdown:.2%}")
    print(f"🔹 Höchster offener Gewinn:  {unrealized.max():.2f}")
    print(f"🔹 Höchster offener Verlust: {unrealized.min():.2f}")


def evaluate_portfolio(all_trades):
    # 1. Kombinierte tägliche Returns (Durchschnitt über Paare)
    all_returns = pd.DataFrame({
        pair: df["Return"] for pair, df in all_trades.items()
    })
    portfolio_return = all_returns.mean(axis=1)

    # 2. Kumulative Rendite
    cumulative_return = (1 + portfolio_return).cumprod() - 1

    # 3. APR & Sharpe
    mean_daily_return = portfolio_return.mean()
    std_daily_return = portfolio_return.std()
    apr = (1 + mean_daily_return) ** 252 - 1
    sharpe = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0
    max_drawdown = (cumulative_return - cumulative_return.cummax()).min()

    print("\n📊 📦 Portfolio Performance (Chan-kompatibel):")
    print(f"🔹 APR:                   {apr:.2%}")
    print(f"🔹 Sharpe Ratio:          {sharpe:.2f}")
    print(f"🔹 Max. Drawdown:         {max_drawdown:.2%}")

    # 4. Plot
    plt.figure(figsize=(12, 5))
    plt.plot(cumulative_return, label="Portfolio Return")
    plt.title("📦 Kumulative Renditekurve (Portfolio)")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_equity(trades_df, pair):
    trades_df["Equity"].plot(figsize=(12, 5), title=f"Equity Curve – {pair[0]} vs {pair[1]}")
    plt.ylabel("Kapital ($)")
    plt.grid(True)
    plt.show()
    
def plot_zscore(zscore, pair, entry=2.0, exit=0.5):
    plt.figure(figsize=(12, 4))
    plt.plot(zscore, label='z-Score', color='blue')
    plt.axhline(entry, color='red', linestyle='--', label=f'Entry +{entry}')
    plt.axhline(-entry, color='red', linestyle='--', label=f'Entry -{entry}')
    plt.axhline(exit, color='gray', linestyle=':', label=f'Exit +{exit}')
    plt.axhline(-exit, color='gray', linestyle=':', label=f'Exit -{exit}')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"📐 z-Score Verlauf – {pair[0]} vs {pair[1]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_cumulative_return(trades_df, pair):
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Cumulative_Return"], label="Cumulative Return")
    plt.title(f"📈 Chan-Style Renditekurve – {pair[0]} vs {pair[1]}")
    plt.ylabel("Rendite (%)")
    plt.xlabel("Datum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# 🔁 Hauptprogramm
if __name__ == "__main__":
    if DATA_MODE == "stocks":
        tickers = get_stock_tickers()
        prices_df = load_data(tickers, START_DATE, END_DATE, interval=INTERVAL)
    elif DATA_MODE == "chan_example":
        tickers = get_chan_example_tickers()
        prices_df = load_data(tickers, START_DATE, END_DATE, interval=INTERVAL)
    elif DATA_MODE == "crypto":
        tickers = get_crypto_tickers(n=1)
        prices_df = load_data(tickers, interval=INTERVAL)
    else:
        raise ValueError("Ungültiger DATA_MODE in config.py")
    
    
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
        p1 = prices_df[pair[0]]
        p2 = prices_df[pair[1]]

        beta_series = calculate_rolling_beta(p1, p2, LOOKBACK_PERIOD)
        spread = p1 - beta_series * p2

        positions = generate_positions_from_zscore(spread, LOOKBACK_PERIOD)

        trades_df = simulate_trades(prices_df, pair, positions, beta_series, initial_capital=INITIAL_CAPITAL)
        evaluate_performance(trades_df)
        plot_equity(trades_df, pair)
        plot_cumulative_return(trades_df, pair)
        all_trades[pair] = trades_df

    evaluate_portfolio(all_trades)
