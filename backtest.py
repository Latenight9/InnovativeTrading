
import pandas as pd
import numpy as np

# 🔁 Importiere deine Analysefunktionen (Dateiname ggf. anpassen)
from Analysis import (
    load_data,
    calculate_top_correlations,
    run_johansen_tests,
    check_spread_stationarity
)

# 📐 Z-Score Berechnung
def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Berechnet den Z-Score des Spreads über ein gleitendes Fenster.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

# 🟢 Hauptprogramm
if __name__ == "__main__":
    # Zeitraum festlegen
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    print("📥 Lade Preisdaten ...")
    prices_df = load_data(start_date, end_date)

    print("🔍 Berechne Korrelationen ...")
    top_pairs = calculate_top_correlations(prices_df, top_n=15)

    print("📊 Führe Johansen-Tests durch ...")
    johansen_results = run_johansen_tests(top_pairs, prices_df)

    print("🧪 Überprüfe auf stationäre Spreads (ADF-Test) ...")
    stationary_pairs = check_spread_stationarity(prices_df, johansen_results)

    if not stationary_pairs:
        print("⚠️ Keine stationären Paare gefunden – beende Programm.")
        exit()

    # 📌 Erstes verfügbares Paar auswählen
    selected_pair = list(stationary_pairs.keys())[0]
    spread = stationary_pairs[selected_pair]["Spread"]

    print(f"✅ Verwende Paar: {selected_pair}")

    # 🧠 Z-Score berechnen
    zscore_series = calculate_zscore(spread, window=20)

    # 📋 Ausgabe zur Kontrolle
    print("\n📉 Z-Score der letzten 5 Zeitpunkte:")
    print(zscore_series.tail())
