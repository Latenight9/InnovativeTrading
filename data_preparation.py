import numpy as np
import pandas as pd

def create_windows(df, window_size, step_size):
    if isinstance(df, pd.DataFrame):
        data = df.values
    else:
        data = df

    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        windows.append(window)
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


def prepare_data(df, window_size, step_size, patch_size=6):
    windows = create_windows(df, window_size, step_size)
    patched = create_patches(windows, patch_size)
    return patched
