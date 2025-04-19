# config.py

# 📅 Datenzeitraum
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"

# 🕒 Intervall für Daten (z. B. "1h", "1d", "15m")
INTERVAL = "1d"

# 📈 Top korrelierte Paare
TOP_N_PAIRS = 1

# 📊 Anzahl Paare für Portfolio-Backtest
MAX_PAIRS = 1

# 💰 Startkapital je Paar
INITIAL_CAPITAL = 10_000.0

# 🧠 Signalparameter
ZSCORE_ENTRY = 3.5
ZSCORE_EXIT = 0.2

# 📉 ADF-Test Schwelle
ADF_THRESHOLD = 0.1

DATA_MODE = "stocks"
