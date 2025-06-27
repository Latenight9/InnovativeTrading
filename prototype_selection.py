# prototype_selection.py

import torch
import torch.nn.functional as F

def select_topk_prototypes(features, prototypes, k=8):
    """
    Wählt pro Sample die Top-k Prototypen basierend auf Cosine Similarity.

    Args:
        features:   Tensor mit Shape (batch_size, embedding_dim)
        prototypes: Tensor mit Shape (n_prototypes, embedding_dim)
        k:          Anzahl der zu wählenden Prototypen

    Returns:
        topk_indices: Tensor mit Shape (batch_size, k)
    """
    # Normiere Features und Prototypen für Cosine Similarity
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # Cosine Similarity berechnen
    similarity = torch.matmul(features, prototypes.T)  # (batch, n_prototypes)

    # Top-k Prototypen pro Sample auswählen
    topk = torch.topk(similarity, k=k, dim=1)

    return topk.indices  # (batch, k)
