import torch
import random

def jitter(window, sigma=0.01):
    noise = torch.normal(0.0, sigma, size=window.shape).to(window.device)
    return window + noise

def scaling(window, sigma=0.1):
    # Skaliere nur entlang der Feature-Dimension (letzte Achse)
    scale = torch.normal(1.0, sigma, size=(1, 1, window.shape[2])).to(window.device)
    return window * scale

def time_warp(window, max_warp=0.2):
    # Simuliere einfache zeitliche Verzerrung durch zufällige Shifts
    warped = window.clone()
    for i in range(window.shape[2]):  # für jede Variable
        shift = random.randint(-int(max_warp * window.shape[1]), int(max_warp * window.shape[1]))
        warped[:, :, i] = torch.roll(window[:, :, i], shifts=shift, dims=0)
    return warped

def augment_window(window):
    # Wende eine zufällige Kombination von Augmentierungen an
    if random.random() < 0.33:
        window = jitter(window)
    if random.random() < 0.33:
        window = scaling(window)
    if random.random() < 0.33:
        window = time_warp(window)
    return window

def augment_batch(batch_windows):
    # Erwartet: batch_windows.shape = (batch_size, n_patches, patch_size, n_channels)
    augmented = []
    for w in batch_windows:
        aug = augment_window(w)
        augmented.append(aug)
    return torch.stack(augmented)
