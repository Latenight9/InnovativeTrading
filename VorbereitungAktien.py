import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_data(start_date, end_date):
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_df = pd.read_html(sp500_url)[0]
    sp500_tickers = sp500_df["Symbol"].tolist()
    invalid_tickers = ['BF.B', 'BRK.B']
    sp500_tickers = [t for t in sp500_tickers if t not in invalid_tickers]

    closing_prices_df = yf.download(sp500_tickers, start=start_date, end=end_date)["Close"]

    closing_prices_df.ffill(inplace=True)
    closing_prices_df.bfill(inplace=True)
    closing_prices_df.interpolate(method='linear', inplace=True)

    return closing_prices_df

def calculate_top_correlations(prices_df, top_n=15):
    correlation_matrix = prices_df.corr(method="pearson")
    correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlation_pairs = correlation_matrix.unstack().dropna().sort_values(ascending=False)
    correlation_pairs = correlation_pairs[correlation_pairs < 0.9999]
    return correlation_pairs.head(top_n)

def plot_top_pairs(top_pairs):
    plt.figure(figsize=(12, 6))
    plt.barh(
        top_pairs.index.map(lambda x: f"{x[0]} - {x[1]}"),
        top_pairs.values,
        color='green'
    )
    plt.xlabel("Pearson-Korrelationskoeffizient")
    plt.ylabel("Aktienpaare")
    plt.title("Top 15 höhstkorrelierdenen Paare")
    plt.xlim(0.85, 1)
    plt.xticks(np.arange(0.85, 1.1, 0.02))
    plt.gca().invert_yaxis()
    plt.show()

#Johansen Test mittels multiprocessing
def johansen_test(pair, prices_df):
    stock1, stock2 = pair
    prices_matrix = prices_df[[stock1, stock2]]

    try:
        result = coint_johansen(prices_matrix, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, 1]
        is_cointegrated = trace_stat > critical_value
        return (stock1, stock2, trace_stat, critical_value, is_cointegrated)
    except Exception as e:
        return (stock1, stock2, None, None, f"Fehler: {e}")

def run_johansen_tests(top_pairs, prices_df):
    print(f"🧠 CPUs verfügbar: {mp.cpu_count()}")
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [
            executor.submit(johansen_test, pair, prices_df)
            for pair in top_pairs.index
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="📈 Johansen-Tests"):
            results.append(f.result())

    end_time = time.time()
    print(f"✅ Johansen-Tests abgeschlossen in {end_time - start_time:.2f} Sekunden.")

    cointegration_results = {
        (s1, s2): {
            "Trace-Statistik": ts,
            "Kritischer Wert (95%)": cv,
            "Cointegrated": c
        }
        for s1, s2, ts, cv, c in results if ts is not None
    }

    return cointegration_results

def print_results(results_dict):
    print("\n📈 Ergebnisse des Johansen-Tests für Top-Paare:\n")
    for pair, result in results_dict.items():
        print(f"{pair}: Trace-Statistik = {result['Trace-Statistik']:.3f}, "
              f"Kritischer Wert (95%) = {result['Kritischer Wert (95%)']:.3f}, "
              f"Cointegrated: {result['Cointegrated']}")

# 🟢 Hauptprogramm
if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    prices_df = load_data(start_date, end_date)
    top_pairs = calculate_top_correlations(prices_df, top_n=15)
    plot_top_pairs(top_pairs)
    results = run_johansen_tests(top_pairs, prices_df)
    print_results(results)
