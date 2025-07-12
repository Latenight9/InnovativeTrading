import torch
import torch.nn.functional as F


def knowledge_distillation_loss(z, c, z_aug, c_aug):
    """
    KD Loss gemäß Paper (Formel 5):
    L_kd = ||z - c||^2 - log(1 - exp(-||z^a - c^a||^2))
    """
    dist_orig = F.mse_loss(z, c, reduction='none').sum(dim=1)
    dist_aug = F.mse_loss(z_aug, c_aug, reduction='none').sum(dim=1)
    loss = dist_orig - torch.log(1 - torch.exp(-dist_aug) + 1e-8)
    return loss.mean()


def contrastive_loss(c_orig, c_aug):
    """
    Contrastive Loss gemäß Paper (Formel 6):
    L_ce = -cosine_similarity(c_orig, c_aug)
    """
    orig_norm = F.normalize(c_orig, dim=1)
    aug_norm = F.normalize(c_aug, dim=1)
    cos_sim = (orig_norm * aug_norm).sum(dim=1)  # (batch,)
    return -cos_sim.mean()


def total_loss(z, c, z_aug, c_aug, c_orig, c_aug_teacher, lambda_ce=0.5):
    """
    Gesamtverlust gemäß Formel (7):
    L_total = L_kd + λ * L_ce
    """
    l_kd = knowledge_distillation_loss(z, c, z_aug, c_aug)
    l_ce = contrastive_loss(c_orig, c_aug_teacher)
    return l_kd + lambda_ce * l_ce, l_kd, l_ce
