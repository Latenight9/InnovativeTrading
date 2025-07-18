import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

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

def prepare_data(df, window_size, step_size, patch_size=6, train=True, target_dim=None, scaler=None):
    if isinstance(df, pd.DataFrame):
        data = df.values
    else:
        data = df

    if train:
        if scaler is not None:
            raise ValueError("❌ Scaler darf beim Training nicht übergeben werden – er wird neu erstellt.")
        if target_dim is None:
            raise ValueError("target_dim muss beim Training angegeben werden.")
        
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, f"scaler_{target_dim}.pkl")

    else:  # Inferenz
        if scaler is None:
            if target_dim is None:
                raise ValueError("target_dim muss angegeben werden, wenn kein Scaler übergeben wird.")
            scaler_path = f"scaler_{target_dim}.pkl"
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler-Datei '{scaler_path}' nicht gefunden.")
            scaler = joblib.load(scaler_path)
        
        data = scaler.transform(data)

    windows = create_windows(data, window_size, step_size)
    patched = create_patches(windows, patch_size)
    return patched

