import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeAttention(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=False)

    def forward(self, input_embed, proto_embed):
        # Erwartet: (B, N, D, E)
        B, N, D, E = input_embed.shape

        # Flatten zu (B, N*D, E)
        Q = input_embed.view(B, N * D, E)
        K = torch.cat([input_embed, proto_embed], dim=1).view(B, N * D * 2, E)
        V = K.clone()

        # Permute: (B, S, E) → (S, B, E)
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        out, _ = self.attn(Q, K, V)  # (S, B, E)
        out = out.permute(1, 0, 2).view(B, N, D, E)  # zurück zu (B, N, D, E)
        return out


class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim=128, patch_size=6, embedding_dim=128,
                 n_heads=8, intermediate_dim=64, n_prototypes=32, num_patches=4):
        super(StudentNet, self).__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.num_patches = num_patches
        self.output_dim = output_dim

        self.embedding = nn.Linear(patch_size, embedding_dim)
        self.prototype_projection = nn.Linear(patch_size, embedding_dim)

        self.attn = PrototypeAttention(embedding_dim=embedding_dim, n_heads=n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embedding_dim)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.LazyLinear(output_dim)

        self.prototypes = nn.Parameter(torch.randn(input_dim * n_prototypes, patch_size))

    def forward(self, x):
        B, N, P, D = x.shape
        assert D == self.input_dim and P == self.patch_size, \
            f"🔥 Input Error: Erwartet (_, _, {self.patch_size}, {self.input_dim}), aber {x.shape}"

        # 1. Eingabe normalisieren
        x_vec = x.view(B * N * D, P)
        x_normed = F.normalize(x_vec, dim=1)
        x_norm = x_normed.view(B, N, D, P)

        # 2. Prototypen normalisieren
        p_normed = F.normalize(self.prototypes, dim=1)

        # 3. Cosine Similarity & Auswahl der besten Prototypen
        sim = torch.matmul(x_normed, p_normed.T)  # (B*N*D, P) @ (P, D*M)^T = (B*N*D, D*M)
        best_idx = sim.argmax(dim=1)
        selected = p_normed[best_idx].view(B, N, D, P)

        # 4. Embedding
        input_embed = self.embedding(x_normed).view(B, N, D, self.embedding_dim)
        proto_embed = self.prototype_projection(selected.view(-1, P)).view(B, N, D, self.embedding_dim)

        # 5. Attention
        attn_out = self.attn(input_embed, proto_embed)

        # 6. Transformer-Block
        x = self.norm1(input_embed + attn_out)
        x_ffn = self.ffn(x)
        z = self.norm2(x + x_ffn)

        # 7. Flatten + Projektion
        z_flat = z.contiguous().view(B, -1)
        return self.output_layer(z_flat)
