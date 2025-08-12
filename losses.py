import torch
import torch.nn.functional as F

def knowledge_distillation_loss(z, c, z_aug, c_aug):
    dist_orig = F.mse_loss(z, c, reduction='none').sum(dim=1)      # (B,)
    dist_aug  = F.mse_loss(z_aug, c_aug, reduction='none').sum(dim=1)  # (B,)
    log_term = -torch.log1p(-torch.exp(-dist_aug).clamp(max=1 - 1e-12))
    return (dist_orig + log_term).mean()

def contrastive_loss(c, c_aug):
    c1 = F.normalize(c, dim=1)
    c2 = F.normalize(c_aug, dim=1)
    cos_sim = (c1 * c2).sum(dim=1)  # (B,)
    return (-cos_sim).mean()

def total_loss(z, c, z_aug, c_aug, *, lambda_ce: float = 0.5):
    l_kd = knowledge_distillation_loss(z, c, z_aug, c_aug)
    l_ce = contrastive_loss(c, c_aug)
    return l_kd + lambda_ce * l_ce, l_kd, l_ce
