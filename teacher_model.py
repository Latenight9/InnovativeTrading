import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class TeacherNet(nn.Module):
    def __init__(self, input_dim, patch_size=6, n_patches=4, embedding_dim=768, output_dim=128):
        super(TeacherNet, self).__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # === Lineares Patch-Embedding ===
        self.linear_embedding = nn.Linear(patch_size * input_dim, embedding_dim)

        # === GPT2-Konfiguration ===
        
        self.llm = GPT2Model.from_pretrained("gpt2")
        self._freeze_llm()  # wie im Paper: nur LayerNorm & Positional fine-tunen

        # === Flatten + Linear-Projektion auf Output-Dimension ===
        self.seq_len = patch_size * n_patches
        self.flatten_proj = nn.Linear(self.embedding_dim * self.n_patches, output_dim)

    def _freeze_llm(self):
        for name, param in self.llm.named_parameters():
            if 'ln' in name or 'wpe' in name or 'wte' in name:
                param.requires_grad = True  # LayerNorm + Positional Embedding = trainierbar
            else:
                param.requires_grad = False  # Attention & FF bleiben frozen

    def forward(self, x):
        # Eingabe: (B, N, P, D)
        B, N, P, D = x.shape
        assert D == self.input_dim and P == self.patch_size and N == self.n_patches

        x = x.view(B, N, P * D)  # (B, N, P*D)
        x = self.linear_embedding(x)  # (B, N, embedding_dim)


        gpt_out = self.llm(inputs_embeds=x)     # (B, N, E)
        gpt_last = gpt_out.last_hidden_state    # (B, N, E)

        x_flat = gpt_last.contiguous().view(B, -1)
        c = self.flatten_proj(x_flat)           # (B, output_dim)
        return c
