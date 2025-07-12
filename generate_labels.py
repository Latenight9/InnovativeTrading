import os
import numpy as np
import pandas as pd
import ast

# === Pfade ===
CSV_PATH = "data/MSL/labeled_anomalies.csv"
TEST_DIR = "data/MSL/test"
LABEL_OUT_DIR = "data/MSL/test_label"

os.makedirs(LABEL_OUT_DIR, exist_ok=True)

# === CSV einlesen ===
df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    chan_id = row["chan_id"]  # z. B. "A-1"
    file_path = os.path.join(TEST_DIR, f"{chan_id}.npy")

    if not os.path.exists(file_path):
        print(f"⛔ Datei nicht gefunden: {file_path}")
        continue

    # Länge der Serie (falls vorhanden – ansonsten aus Datei lesen)
    if "num_values" in row:
        series_len = int(row["num_values"])
    else:
        series_len = np.load(file_path).shape[0]

    label = np.zeros(series_len, dtype=int)

    # Anomalie-Intervalle
    try:
        intervals = ast.literal_eval(row["anomaly_sequences"])
    except Exception as e:
        print(f"⚠️ Fehler beim Parsen der Intervalle für {chan_id}: {e}")
        continue

    for start, end in intervals:
        # Sicherheit: min/max-Grenzen
        start = max(0, start)
        end = min(series_len - 1, end)
        label[start:end + 1] = 1

    # Speichern
    label_path = os.path.join(LABEL_OUT_DIR, f"{chan_id}.npy")
    np.save(label_path, label)
    print(f"✅ Label gespeichert: {label_path}")
