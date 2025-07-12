import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeAttention(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)

    def forward(self, input_embed, proto_embed):
        B, N, D, E = input_embed.shape
        Q = input_embed.view(B, N * D, E)
        K = torch.cat([input_embed, proto_embed], dim=1)  # (B, 2*N*D, E)
        V = K.clone()
        out, _ = self.attn(Q, K, V)
        return out.view(B, N, D, E)


class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim=128, patch_size=6, embedding_dim=128,
                 n_heads=8, intermediate_dim=64, n_prototypes=32, num_patches=4):
        super(StudentNet, self).__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.num_patches = num_patches
        self.output_dim = output_dim  # für dynamische Init, falls nötig

        self.input_norm = nn.InstanceNorm1d(self.patch_size, affine=False)

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

        # 🔹 LazyLinear = automatische Anpassung bei erstem Input
        self.output_layer = nn.LazyLinear(output_dim)

        # 🔹 Prototypen: (D * M, P)
        self.prototypes = nn.Parameter(torch.randn(input_dim * n_prototypes, patch_size))

    def forward(self, x):
        B, N, P, D = x.shape
        assert D == self.input_dim and P == self.patch_size, \
            f"🔥 Input Error: Erwartet (_, _, {self.patch_size}, {self.input_dim}), aber {x.shape}"

        # 1. Input-Normalisierung (pro Channel)
        x_flat = x.permute(0, 1, 3, 2).reshape(B * N * D, P)
        x_flat = self.input_norm(x_flat.unsqueeze(1)).squeeze(1)

        # 2. Prototypen-Normalisierung
        protos = self.input_norm(self.prototypes.unsqueeze(1)).squeeze(1)

        # 3. Cosine Similarity → beste Prototypen auswählen
        x_norm = F.normalize(x_flat, dim=1)
        p_norm = F.normalize(protos, dim=1)
        sim = torch.matmul(x_norm, p_norm.T)
        best_idx = sim.argmax(dim=1)
        selected = protos[best_idx].view(B, N, D, P)

        # 4. Embedding
        input_embed = self.embedding(x_flat).view(B, N, D, self.embedding_dim)
        proto_embed = self.prototype_projection(selected.view(-1, P)).view(B, N, D, self.embedding_dim)

        # 5. Multi-Head Attention mit Prototypen
        out = self.attn(input_embed, proto_embed)

        # 6. Transformer-artiger Block
        x = self.norm1(input_embed + out)
        x_ffn = self.ffn(x)
        z = self.norm2(x + x_ffn)

        # 7. Flatten + Projektion
        z_flat = z.contiguous().view(B, -1)
        return self.output_layer(z_flat)
