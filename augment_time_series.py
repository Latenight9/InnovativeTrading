import torch

# — Realistischere Augment-Defaults (paper-konform, nicht zu aggressiv) —
DEFAULTS = dict(
    jitter_sigma=0.05,
    scaling_sigma=0.10,
    max_warp=0.30,   # Anteil der Segmentlänge
    p_jitter=0.50,
    p_scaling=0.35,
    p_warp=0.15,
)

def jitter(x, sigma=0.07):
    # x: (L, D)
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=0.12):
    # x: (L, D)
    L, D = x.shape
    scale = (torch.randn(1, D, device=x.device) * sigma) + 1.0  # (1, D)
    return x * scale

def time_warp(x, max_warp=0.4):
    # x: (L, D). Verschiebt je Kanal zyklisch um bis zu floor(max_warp * L) Schritte.
    L, D = x.shape
    x_warped = x.clone()
    max_shift = int(max(1, max_warp * L))
    shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(D,), device=x.device)
    for d in range(D):
        x_warped[:, d] = torch.roll(x[:, d], shifts=int(shifts[d].item()), dims=0)
    return x_warped

def _sample_aug_type(p_jitter, p_scaling, p_warp):
    # wählt eine Augmentierung gemäß Gewichten
    probs = torch.tensor([p_jitter, p_scaling, p_warp], dtype=torch.float)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, 1).item()
    return idx  # 0=jitter,1=scaling,2=warp

def augment_batch(batch_windows, segment_ratio=1/3, **kwargs):
    cfg = {**DEFAULTS, **kwargs}
    B, N, P, D = batch_windows.shape
    total_length = N * P
    segment_len = max(1, int(total_length * segment_ratio))

    augmented = batch_windows.clone()
    for i in range(B):
        # flach über Zeit, damit segmentübergreifend augmentiert werden kann
        flat = batch_windows[i].contiguous().view(-1, D)  # (N*P, D)

        # Segment-Start
        start = torch.randint(0, total_length - segment_len + 1, (1,)).item()
        end = start + segment_len
        segment = flat[start:end].clone()  # (L, D)

        # Augmentierungs-Typ wählen (gewichtet)
        a = _sample_aug_type(cfg["p_jitter"], cfg["p_scaling"], cfg["p_warp"])
        if a == 0:
            segment = jitter(segment, sigma=cfg["jitter_sigma"])
        elif a == 1:
            segment = scaling(segment, sigma=cfg["scaling_sigma"])
        else:
            segment = time_warp(segment, max_warp=cfg["max_warp"])

        # zurückschreiben
        flat[start:end] = segment
        augmented[i] = flat.contiguous().view(N, P, D)

    return augmented
