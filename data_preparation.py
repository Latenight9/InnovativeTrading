import numpy as np
import pandas as pd

def create_windows(df, window_size=24, step_size=1):
    data = df.values
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def normalize_windows(windows):
    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True) + 1e-8
    normalized = (windows - means) / stds
    return normalized

def create_patches(windows, patch_size=6):
    """
    Unterteilt jedes Fenster in kleinere Patches.
    z.B. window_size=24, patch_size=6 → 4 Patches pro Fenster.
    """


    patches = []
    for w in windows:
        window_patches = []
        for i in range(0, len(w), patch_size):
            patch = w[i:i + patch_size]
            window_patches.append(patch)
        patches.append(window_patches)
    return np.array(patches)

def prepare_data(df, window_size=24, step_size=1, patch_size=6):
    """
    Pipeline: Fensterung → Normalisierung → Patchbildung.
    Gibt Patches der Form (n_windows, n_patches, patch_size, n_assets) zurück.
    """
    windows = create_windows(df, window_size, step_size)
    normalized = normalize_windows(windows)
    patches = create_patches(normalized, patch_size)
    return patches
