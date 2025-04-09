# config.py

# 📅 Datenzeitraum
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"

# 🕒 Intervall für Daten (z. B. "1h", "1d", "15m")
INTERVAL = "1h"

# 📈 Top korrelierte Paare
TOP_N_PAIRS = 15

# 📊 Anzahl Paare für Portfolio-Backtest
MAX_PAIRS = 5

# 💰 Startkapital je Paar
INITIAL_CAPITAL = 10_000.0

# 🧠 Signalparameter
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5

# 📉 ADF-Test Schwelle
ADF_THRESHOLD = 0.05

DATA_MODE = "crypto"
