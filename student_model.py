import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim=128, patch_size=6, embedding_dim=128,
                 n_heads=8, intermediate_dim=64, n_prototypes=32, num_patches=4):
        super(StudentNet, self).__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.num_patches = num_patches

        self.patch_norm = nn.LayerNorm(patch_size)
        self.proto_norm = nn.LayerNorm(patch_size)

        self.embedding = nn.Linear(patch_size, embedding_dim)
        self.prototype_projection = nn.Linear(patch_size, embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embedding_dim)
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim * num_patches * input_dim, output_dim)

        # (D * M, P)
        self.prototypes = nn.Parameter(torch.randn(input_dim * n_prototypes, patch_size))

    def forward(self, x):
        B, N, P, D = x.shape  # (Batch, Num_Patches, Patch_Size, Features)
        assert D == self.input_dim and P == self.patch_size, \
        f"🔥 Input Error: Erwartet (_, _, {self.patch_size}, {self.input_dim}), aber {x.shape}"

        # === 🔹1. Input-Normalisierung pro Patch (feature-wise) ===
        x = x.permute(0, 1, 3, 2)       # (B, N, D, P)
        x = self.patch_norm(x)          # LayerNorm über P (letzte Achse)
        x = x.permute(0, 1, 3, 2)       # zurück zu (B, N, P, D)


        # === 🔹2. Input flatten für Projektionen ===
        x_flat = x.permute(0, 1, 3, 2).reshape(B * N * D, P)  # (B*N*D, P)

        # === 🔹3. Prototypen vorbereiten ===
        protos = self.proto_norm(self.prototypes)  # (D*M, P)

        # === 🔹4. Cosine-Similarity für Prototypenwahl ===
        x_norm = F.normalize(x_flat, dim=1)  # (B*N*D, P)
        p_norm = F.normalize(protos, dim=1)  # (D*M, P)
        sim = torch.matmul(x_norm, p_norm.T)  # (B*N*D, D*M)
        best_idx = sim.argmax(dim=1)         # (B*N*D,)
        selected = protos[best_idx].view(B, N, D, P)  # (B, N, D, P)

        # === 🔹5. Patch- & Prototyp-Embedding ===
        input_embed = self.embedding(x_flat).view(B, N, D, self.embedding_dim)
        proto_embed = self.prototype_projection(selected.view(-1, P)).view(B, N, D, self.embedding_dim)

        # === 🔹6. Formel (3): Gewichtung (attention-ähnlich) ===
        qw = F.normalize(input_embed, dim=-1)
        kw = F.normalize(input_embed, dim=-1)
        km = F.normalize(proto_embed, dim=-1)

        dot_kw = torch.sum(qw * kw, dim=-1, keepdim=True)  # (B, N, D, 1)
        dot_km = torch.sum(qw * km, dim=-1, keepdim=True)  # (B, N, D, 1)

        all_scores = torch.cat([dot_kw, dot_km], dim=-1)  # (B, N, D, 2)
        weights = F.softmax(all_scores, dim=-1)
        sw, sm = weights[..., 0:1], weights[..., 1:2]

        # === 🔹7. Formel (4): Lineare Kombination ===
        out = sw * kw + sm * km  # (B, N, D, E)

        # === 🔹8. Transformer-like Block ===
        x = self.norm1(input_embed + out)
        x_ffn = self.ffn(x)
        z = self.norm2(x + x_ffn)  # (B, N, D, E)

        # === 🔹9. Flatten + Projektion ===
        z_flat = z.contiguous().view(B, -1)
        assert z_flat.shape[1] == self.output_layer.in_features, \
        f"🔥 Dimension Error: {z_flat.shape[1]} != {self.output_layer.in_features}"
        return self.output_layer(z_flat) # (B, output_dim)
