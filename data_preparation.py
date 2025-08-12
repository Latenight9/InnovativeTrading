import numpy as np
import pandas as pd

def create_windows(data, window_size, step_size):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=np.float32)
    T, D = data.shape
    if T < window_size:
        return np.empty((0, window_size, D), dtype=np.float32)

    windows = []
    for start in range(0, T - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, D), dtype=np.float32)

def instance_normalize_windows(windows, eps: float = 1e-5):
    if windows.size == 0:
        return windows
    mean = windows.mean(axis=1, keepdims=True)  # (N, 1, D)
    std  = windows.std(axis=1, keepdims=True)   # (N, 1, D)
    return (windows - mean) / (std + eps)

def create_patches(windows, patch_size=6):
    if windows.size == 0:
        return np.empty((0, 0, patch_size, 0), dtype=np.float32)
    N, W, D = windows.shape
    assert W % patch_size == 0 
    n_patches = W // patch_size
    patched = windows.reshape(N, n_patches, patch_size, D).astype(np.float32)
    return patched

def prepare_data(df, window_size, step_size, patch_size=6, train=True, target_dim=None, scaler=None):
    windows = create_windows(df, window_size, step_size)
    windows = instance_normalize_windows(windows)
    patched = create_patches(windows, patch_size)
    return patched
