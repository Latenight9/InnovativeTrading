import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, intermediate_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim=128, patch_size=6, embedding_dim=128,
                 n_heads=8, intermediate_dim=64, n_prototypes=32, num_patches=4,
                 n_layers=2):
        super().__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.output_dim = output_dim

        self.embedding = nn.Linear(patch_size, embedding_dim)
        self.prototype_projection = nn.Linear(patch_size, embedding_dim)

        self.prototypes = nn.Parameter(torch.randn(D, 32, patch_size))
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, max_len=num_patches * input_dim)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, n_heads, intermediate_dim)
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(num_patches * input_dim * embedding_dim, output_dim)

    def forward(self, x):
        B, N, P, D = x.shape
        assert D == self.input_dim and P == self.patch_size

        x_flat = x.view(B * N * D, P)
        x_normed = F.normalize(x_flat, dim=1)
        x_norm = x_normed.view(B, N, D, P)

        p_normed = F.normalize(self.prototypes, dim=2)  # (D, 32, P)

        selected = torch.zeros_like(x_norm)

        for d in range(D):
            x_d = x_norm[:, :, d, :]                 # (B, N, P)
            p_d = p_normed[d]                        # (32, P)
            sim = torch.matmul(x_d, p_d.T)           # (B, N, 32)
            best_idx = sim.argmax(dim=2)             # (B, N)

            # → Performante Auswahl via gather
            p_d_exp = p_d.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)        # (B, N, 32, P)
            best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, P)  # (B, N, 1, P)
            selected_d = torch.gather(p_d_exp, 2, best_idx_exp).squeeze(2)     # (B, N, P)

            selected[:, :, d, :] = selected_d

        input_embed = self.embedding(x_flat).view(B, N, D, self.embedding_dim)
        proto_embed = self.prototype_projection(selected.view(B * N * D, P)).view(B, N, D, self.embedding_dim)

        combined = input_embed + proto_embed
        seq = combined.view(B, -1, self.embedding_dim)

        seq = self.pos_enc(seq)
        seq = self.transformer_blocks(seq)

        z = seq.contiguous().view(B, -1)
        return self.output_layer(z)

