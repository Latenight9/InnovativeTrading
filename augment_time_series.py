import numpy as np
import torch
import torch.nn.functional as F


def time_warp(window, sigma=0.2):
    """
    Verzerrt die Zeitachse nichtlinear (z. B. durch Interpolation).
    """
    T, D = window.shape
    device = window.device

    # Zufällige Verzerrung der Zeitachsen-Positionen
    original_idx = torch.arange(T, device=device).float()
    random_offsets = torch.normal(0.0, sigma, size=(T,), device=device)
    warped_idx = original_idx + random_offsets
    warped_idx = torch.clamp(warped_idx, 0, T - 1)

    # Interpolieren entlang Zeitachse (pro Channel separat)
    warped = F.interpolate(
        window.T.unsqueeze(0),  # (1, D, T)
        size=T,
        mode='linear',
        align_corners=True,
        recompute_scale_factor=False
    ).squeeze(0).T

    return warped


def augment_window(window, noise_std=0.01, scale_range=(0.95, 1.05), use_warping=True):
    """
    Kombinierte Augmentierung: scaling + jittering + optional warping
    """
    # Scaling
    scale = np.random.uniform(*scale_range, size=(1, window.shape[1]))
    scale = torch.tensor(scale, dtype=window.dtype, device=window.device)
    window = window * scale

    # Jittering
    noise = torch.randn_like(window) * noise_std
    window = window + noise

    # Optional: Warping
    if use_warping:
        window = time_warp(window)

    return window


def augment_batch(batch_windows):
    return torch.stack([augment_window(w) for w in batch_windows])
