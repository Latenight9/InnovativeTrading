import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size=6, embedding_dim=128, n_heads=8, intermediate_dim=64, n_prototypes=32):
        super(StudentNet, self).__init__()

        # Lineares Embedding pro Patch
        self.embedding = nn.Linear(patch_size * input_dim, embedding_dim)

        # Cross-Attention: Q = Input, K/V = Prototypen
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)

        # Feedforward Layer nach Attention
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embedding_dim)
        )

        # Trainierbare Prototypen (werden später per Auswahl gefiltert)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))

    def forward(self, x, selected_proto_ids=None):
        """
        x: Tensor (batch, n_patches, patch_size, input_dim)
        selected_proto_ids: Tensor (batch, top_k), optional – Indexe der Prototypen
        """

        batch_size, n_patches, patch_size, input_dim = x.shape

        # ⬛ 1. Flatten der Patches und lineares Embedding
        x = x.reshape(batch_size, n_patches, patch_size * input_dim)
        x = self.embedding(x)  # (batch, n_patches, embedding_dim)

        # ⬛ 2. Prototypen-Auswahl vorbereiten
        if selected_proto_ids is None:
            # Fallback: alle Prototypen verwenden
            prototypes = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_proto, emb)
        else:
            # Wähle pro Sample die Top-k Prototypen aus
            prototypes = []
            for b in range(batch_size):
                selected = self.prototypes[selected_proto_ids[b]]  # (top_k, emb)
                prototypes.append(selected)
            prototypes = torch.stack(prototypes, dim=0)  # (batch, top_k, emb)

        # ⬛ 3. Cross-Attention: Q = x (Input), K/V = Prototypen
        attn_output, _ = self.attention(query=x, key=prototypes, value=prototypes)  # (batch, n_patches, emb)

        # ⬛ 4. Feedforward-Netzwerk
        x = self.ffn(attn_output)  # (batch, n_patches, emb)

        # ⬛ 5. Aggregiere Patches zu Feature-Vektor (letzter Token)
        features = x[:, -1, :]  # (batch, emb_dim)

        # ⬛ 6. Ähnlichkeit zu allen Prototypen (für optionales Logging oder Anomalie-Ranking)
        proto_norm = F.normalize(self.prototypes, dim=1)      # (n_proto, emb)
        feat_norm = F.normalize(features, dim=1)              # (batch, emb)
        sim = torch.matmul(feat_norm, proto_norm.T)           # (batch, n_proto)

        return features, sim
