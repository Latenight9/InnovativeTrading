import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 📁 Wo der Scaler gespeichert/geladen wird
SCALER_PATH = "scaler.pkl"

def create_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def create_patches(windows, patch_size=6):
    n_windows, window_size, n_assets = windows.shape
    patches = []
    for w in windows:
        window_patches = []
        for i in range(0, window_size, patch_size):
            patch = w[i:i + patch_size]
            window_patches.append(patch)
        patches.append(window_patches)
    return np.array(patches)

def prepare_data(df, window_size, step_size, patch_size=6, train=True):
    if isinstance(df, pd.DataFrame):
        data = df.values
    else:
        data = df

    # 🔍 Skalierung
    if train:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, SCALER_PATH)  # 💾 Speichern für später
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler-Datei nicht gefunden: erst Training durchführen!")
        scaler = joblib.load(SCALER_PATH)
        data = scaler.transform(data)

    windows = create_windows(data, window_size, step_size)
    patched = create_patches(windows, patch_size)
    return patched
