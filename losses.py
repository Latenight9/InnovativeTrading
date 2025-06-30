import torch
import torch.nn.functional as F

def knowledge_distillation_loss(student_out, teacher_out, labels):
    """
    Berechnet den KD-Loss zwischen Student- und Teacher-Ausgabe (Repräsentationen).
    Entspricht Formel (5) im Paper.
    
    ℓ(z, c) = exp(-‖z - c‖²)
    L_kd = - (1 - y) * log(ℓ) - y * log(1 - ℓ)
    
    Args:
        student_out: Tensor (batch, emb_dim)
        teacher_out: Tensor (batch, emb_dim)
        labels: Tensor (batch,) – 0 = normal, 1 = anomal
    
    Returns:
        Scalar loss
    """
    # Quadratdifferenz
    mse = F.mse_loss(student_out, teacher_out, reduction='none')  # (batch, emb_dim)
    dist = mse.sum(dim=1)  # ‖z - c‖² → (batch,)

    sim = torch.exp(-dist)  # ℓ(z, c) = exp(−‖z - c‖²)

    # Binary Log Loss entsprechend Ground Truth
    loss = - (1 - labels) * torch.log(sim + 1e-8) \
           - labels * torch.log(1 - sim + 1e-8)

    return loss.mean()


def contrastive_loss(original_feat, augmented_feat):
    """
    Vergleich zwischen Original- und Augmentiertem Feature (beide vom Student).
    Entspricht Formel (6) im Paper:
    L_ce = 1 - cos(z, z_a)
    
    Args:
        original_feat: Tensor (batch, emb_dim)
        augmented_feat: Tensor (batch, emb_dim)
    
    Returns:
        Scalar loss
    """
    # Cosine Similarity
    original_feat = F.normalize(original_feat, dim=1)
    augmented_feat = F.normalize(augmented_feat, dim=1)

    cos_sim = (original_feat * augmented_feat).sum(dim=1)  # (batch,)
    return (1 - cos_sim).mean()  # Je höher cos_sim, desto niedriger der Verlust


def total_loss(student_out, teacher_out, student_aug_out, labels, lambda_ce=1.0):
    """
    Kombinierter Loss aus KD und Contrastive Loss.

    Args:
        student_out: Originalausgabe des Student
        teacher_out: Ausgabe des Teacher
        student_aug_out: Student-Ausgabe für augmentiertes Fenster
        labels: Ground truth labels (0 = normal, 1 = anomal)
        lambda_ce: Gewichtung für den Contrastive Loss

    Returns:
        Gesamtverlust, Einzelverluste
    """
    l_kd = knowledge_distillation_loss(student_out, teacher_out, labels)
    l_ce = contrastive_loss(student_out, student_aug_out)
    total = l_kd + lambda_ce * l_ce
    return total, l_kd, l_ce
