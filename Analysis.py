import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import ccxt
from datetime import datetime, timezone, timedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from config import ADF_THRESHOLD, DATA_MODE  # zentrale Schwelle für ADF-Test

# 1️⃣ Daten laden
def load_data(tickers, start_date=None, end_date=None, interval="1h", since_days=100):
    if DATA_MODE in ["stocks", "chan_example"]:
        print("⬇️ Lade Preisdaten von Yahoo Finance...")
        df = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=False)["Close"]
    else:
        print("⬇️ Lade Preisdaten von Binance über ccxt...")
        exchange = ccxt.binance()
        end_time = datetime.now(timezone.utc)
        since_time = int((end_time - timedelta(days=since_days)).timestamp() * 1000)

        all_data = {}
        for ticker in tickers:
            try:
                ohlcv = exchange.fetch_ohlcv(ticker, timeframe=interval, since=since_time)
                df_ohlcv = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], unit='ms')
                df_ohlcv.set_index('timestamp', inplace=True)
                all_data[ticker] = df_ohlcv['close']
            except Exception as e:
                print(f"❌ Fehler bei {ticker}: {e}")
        df = pd.DataFrame(all_data)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.interpolate(method='linear', inplace=True)
    print("📊 Anzahl Zeitpunkte (Zeilen):", len(df))
    return df


# 2️⃣ Korrelation berechnen
def calculate_top_correlations(prices_df, top_n):
    correlation_matrix = prices_df.corr(method="pearson")
    correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlation_pairs = correlation_matrix.unstack().dropna().sort_values(ascending=False)
    correlation_pairs = correlation_pairs[correlation_pairs < 0.9999]
    return correlation_pairs.head(top_n)

# 🔍 Korrelation visualisieren
def plot_top_pairs(top_pairs):
    plt.figure(figsize=(12, 6))
    plt.barh(
        top_pairs.index.map(lambda x: f"{x[0]} - {x[1]}"),
        top_pairs.values,
        color='green'
    )
    plt.xlabel("Pearson-Korrelationskoeffizient")
    plt.ylabel("Aktienpaare")
    plt.title("Top 15 höchsten Korrelationen zwischen Aktien")
    plt.xlim(0.85, 1)
    plt.xticks(np.arange(0.85, 1.1, 0.02))
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

# 3️⃣ Johansen-Test für ein Paar
def johansen_test(pair, prices_df):
    stock1, stock2 = pair
    
    try:
        log_prices = np.log(prices_df[[stock1, stock2]])
        result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, 1]
        is_cointegrated = trace_stat > critical_value
        eigenvector = result.evec[:, 0]
        return (stock1, stock2, trace_stat, critical_value, is_cointegrated, eigenvector)
    except Exception as e:
        return (stock1, stock2, None, None, False, f"Fehler: {e}")

# 4️⃣ Johansen-Test parallelisiert für viele Paare
def run_johansen_tests(top_pairs, prices_df):
    print(f"🧠 CPUs verfügbar: {mp.cpu_count()}")
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(johansen_test, pair, prices_df) for pair in top_pairs.index]
        for f in tqdm(as_completed(futures), total=len(futures), desc="📈 Johansen-Tests"):
            results.append(f.result())

    end_time = time.time()
    print(f"✅ Johansen-Tests abgeschlossen in {end_time - start_time:.2f} Sekunden.")

    cointegration_results = {
        (s1, s2): {
            "Trace-Statistik": ts,
            "Kritischer Wert (95%)": cv,
            "Cointegrated": c,
            "Eigenvektor": vec
        }
        for s1, s2, ts, cv, c, vec in results if ts is not None
    }

    return cointegration_results

# 5️⃣ ADF-Test für stationären Spread
def check_spread_stationarity(prices_df, cointegration_results):
    print("\n🔍 Stationarität des Spreads (ADF-Test):\n")
    stationary_pairs = {}

    for (s1, s2), result in cointegration_results.items():
        if isinstance(result["Eigenvektor"], str):
            continue

        vec = result["Eigenvektor"]
        if abs(vec[0]) < 1e-6:
            print(f"⚠️ Eigenvektor für {s1}-{s2} instabil (vec[0] ≈ 0), übersprungen.")
            continue

        log_p1 = np.log(prices_df[s1])
        log_p2 = np.log(prices_df[s2])
        
        beta = -vec[1] / vec[0]
        #beta = np.polyfit(log_p2, log_p1, 1)[0] 

        # Schutz gegen unsinnige Verhältnisse
        # if abs(beta) > 10 or abs(beta) < 0.01:
        #     print(f"⚠️ Unplausibles Beta-Verhältnis {beta:.4f} bei {s1}-{s2} → übersprungen.")
        #     continue

        
        spread = log_p1 - beta * log_p2

        adf_result = adfuller(spread)
        p_value = adf_result[1]
        is_stationary = p_value < ADF_THRESHOLD

        print(f"{s1} - {s2}: ADF p-value = {p_value:.4f} → Stationär: {is_stationary} | beta = {beta:.4f}")

        if is_stationary:
            stationary_pairs[(s1, s2)] = {
                **result,
                "ADF p-value": p_value,
                "Spread": spread,
                "Beta": beta
            }

    return stationary_pairs

# 🖨️ Ausgabe
def print_results(results_dict):
    print("\n📈 Ergebnisse des Johansen-Tests für Top-Paare:\n")
    for pair, result in results_dict.items():
        print(f"{pair}: Trace-Statistik = {result['Trace-Statistik']:.3f}, "
              f"Kritischer Wert (95%) = {result['Kritischer Wert (95%)']:.3f}, "
              f"Cointegrated: {result['Cointegrated']}")
