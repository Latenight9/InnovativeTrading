import torch
import numpy as np

@torch.no_grad()
def l2_scores(student_out, teacher_out):
    if student_out.dim() != 2:
        student_out = student_out.view(student_out.size(0), -1)
    if teacher_out.dim() != 2:
        teacher_out = teacher_out.view(teacher_out.size(0), -1)
    diff = student_out - teacher_out
    return (diff * diff).sum(dim=1)  

def normalize_scores(scores: np.ndarray):
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-8)
