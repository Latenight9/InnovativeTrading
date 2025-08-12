# student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,E)

    def forward(self, x: torch.Tensor):
        # x: (B, T, E)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ProtoAwareBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, hidden, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, proto_tokens: torch.Tensor):
        # q_tokens: (B,T,E), proto_tokens: (B,T,E)
        mem = torch.cat([q_tokens, proto_tokens], dim=1)        # (B,2T,E)
        attn_out, _ = self.attn(q_tokens, mem, mem, need_weights=False)
        x = self.norm1(q_tokens + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x

class StudentNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_prototypes: int = 32,
        patch_size: int = 6,
        num_patches: int = 4,
        embedding_dim: int = 128,
        output_dim: int = 128,
        n_heads: int = 8,
        intermediate_dim: int = 64,
        dropout: float = 0.1,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim     = input_dim
        self.n_prototypes  = n_prototypes
        self.patch_size    = patch_size
        self.num_patches   = num_patches
        self.embedding_dim = embedding_dim
        self.output_dim    = output_dim

        P, D, M, E = patch_size, input_dim, n_prototypes, embedding_dim

        # Prototypen im Patch-Raum: (D, M, P)
        self.prototypes = nn.Parameter(torch.randn(D, M, P) * 0.02)

        # Lineare Projektionen Patch->Embedding
        self.input_proj = nn.Linear(P, E)
        self.proto_proj = nn.Linear(P, E)

        # Positional Encoding (wie zuvor; Key: pos_enc.pe)
        self.pos_enc = PositionalEncoding(E, max_len=num_patches * input_dim + 8)

        # Prototype-aware Encoder-Blocks (Key-Prefix: blocks.*)
        self.blocks = nn.ModuleList([
            ProtoAwareBlock(E, n_heads=n_heads, hidden=intermediate_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Flatten + Linear Head (kein Pooling)
        self.flat_proj = nn.Linear(E * (num_patches * input_dim), output_dim)

    # ---- Harte Proto-Selektion: Argmax im *Embedding*-Raum, Gather im Patch-Raum ----
    def _hard_select_prototypes(self, x_bndp: torch.Tensor) -> torch.Tensor:
   
        B, N, D, P = x_bndp.shape
        E, M = self.embedding_dim, self.n_prototypes

    # Input-Patches → Embedding-Tokens: (B,N,D,E), dann *Window-Aggregation* über N
        x_tok = self.input_proj(x_bndp.reshape(B * N * D, P)).view(B, N, D, E)
        x_tok = F.normalize(x_tok, dim=-1)
        x_win = F.normalize(x_tok.amax(dim=1), dim=-1)           # (B,D,E)

    # Proto-Patches → Embedding-Tokens
        p_tok = self.proto_proj(self.prototypes.view(D * M, P)).view(D, M, E)
        p_tok = F.normalize(p_tok, dim=-1)                       # (D,M,E)

    # Ähnlichkeit (B,D,M) & harte Auswahl je (B,D)
        sim = torch.einsum("bde,dme->bdm", x_win, p_tok)         # (B,D,M)
        idx = sim.argmax(dim=-1)                                 # (B,D)

    # Im *Patch*-Raum gathern: (D,M,P) -> (B,D,P), dann auf N broadcasten
        p_norm  = F.normalize(self.prototypes, p=2, dim=-1)      # (D,M,P)
        p_exp   = p_norm.unsqueeze(0).expand(B, D, M, P)         # (B,D,M,P)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, D, 1, P)
        selected_bdP = torch.gather(p_exp, dim=2, index=idx_exp).squeeze(2)  # (B,D,P)
        selected = selected_bdP.unsqueeze(1).expand(B, N, D, P)              # (B,N,D,P)
        return selected

    # -------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, P, D)
        """
        if x.dim() != 4:
            raise ValueError("StudentNet expects input of shape (B, N, P, D)")
        B, N, P, D = x.shape
        assert D == self.input_dim and P == self.patch_size and N == self.num_patches, "Shape mismatch"

        # (B,N,P,D) -> (B,N,D,P)
        x_bndp = x.permute(0, 1, 3, 2).contiguous()

        # Harte Prototypenwahl pro (B,N,D) (Argmax im Embedding-Raum)
        selected = self._hard_select_prototypes(x_bndp)  # (B,N,D,P)

        # Tokens projizieren: Patch->E und auf Sequenz flachziehen (T = N*D)
        x_tokens = self.input_proj(x_bndp.reshape(B * N * D, P)).view(B, N * D, self.embedding_dim)
        p_tokens = self.proto_proj(selected.reshape(B * N * D, P)).view(B, N * D, self.embedding_dim)

        # Positional Encoding auf Input-Tokens (wie früher)
        h = self.pos_enc(x_tokens)
        p_tokens = self.pos_enc(p_tokens)
        
        # Prototype-aware Encoder
        for block in self.blocks:
            h = block(h, p_tokens)  # (B, T, E)

        # Flatten + Linear Head
        z = self.flat_proj(h.reshape(B, -1))  # (B, output_dim)
        return z
