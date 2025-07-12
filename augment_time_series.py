import torch

def jitter(x, sigma=0.4):
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=0.5):
    _, P, D = x.shape
    scale = torch.randn(1, 1, D, device=x.device) * sigma + 1.0
    return x * scale

def time_warp(x, max_warp=0.7):
    _, P, D = x.shape
    x_warped = x.clone()
    shifts = ((2 * torch.rand(1, D, device=x.device) - 1) * int(max_warp * P)).round().int()
    for d in range(D):
        x_warped[0, :, d] = torch.roll(x[0, :, d], shifts=shifts[0, d].item(), dims=0)
    return x_warped

def augment_batch(batch_windows, segment_ratio=1/3):
    """
    Augmentiert pro Sample ein zufälliges Segment mit Länge ≈ segment_ratio * WINDOW_SIZE
    """
    B, N, P, D = batch_windows.shape  # z. B. (B, 4, 6, D)
    total_length = N * P             # z. B. 24
    segment_len = max(1, int(total_length * segment_ratio))

    augmented = batch_windows.clone()
    for i in range(B):
        # --- Flache Zeitreihe für Zugriff über Fenstergrenzen hinweg ---
        flat = batch_windows[i].view(-1, D)  # (N*P, D)

        # --- Segmentposition bestimmen ---
        start = torch.randint(0, total_length - segment_len + 1, (1,)).item()
        end = start + segment_len
        segment = flat[start:end].clone()

        # --- Augmentationstyp wählen ---
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            segment = jitter(segment, sigma=0.4)
        elif aug_type == 1:
            segment = scaling(segment.unsqueeze(0), sigma=0.5).squeeze(0)
        elif aug_type == 2:
            segment = time_warp(segment.unsqueeze(0), max_warp=0.7).squeeze(0)

        # --- Zurückschreiben ---
        flat[start:end] = segment
        augmented[i] = flat.view(N, P, D)

    return augmented

