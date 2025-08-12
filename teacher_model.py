import torch
import torch.nn as nn
from transformers import GPT2Model

class TeacherNet(nn.Module):
    def __init__(self, input_dim, patch_size=6, n_patches=4, embedding_dim=768, output_dim=128, n_heads=8):
        super().__init__()
        self.input_dim   = input_dim
        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.output_dim  = output_dim

        # GPT-2 laden und auf 6 Layer kürzen
        llm = GPT2Model.from_pretrained("gpt2")
        if hasattr(llm, "h"):
            if len(llm.h) > 6:
                llm.h = nn.ModuleList(list(llm.h)[:6])
            llm.config.n_layer = 6
            n_embd = llm.config.n_embd
        elif hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
            if len(llm.transformer.h) > 6:
                llm.transformer.h = nn.ModuleList(list(llm.transformer.h)[:6])
            llm.config.n_layer = 6
            n_embd = llm.config.n_embd
        else:
            raise RuntimeError("Unerwartete GPT-2 Struktur (erwartet `h` oder `transformer.h`).")

        self.llm = llm
        self.n_embd = n_embd

        # Lineares Patch-Embedding -> GPT2 Hidden-Size
        self.linear_embedding = nn.Linear(self.input_dim * self.patch_size, self.n_embd)

        # Flatten + Linear auf Ausgabe c
        self.flatten_proj = nn.Linear(self.n_patches * self.n_embd, self.output_dim)

        # Alles einfrieren, dann nur wpe + LayerNorms trainierbar
        for p in self.llm.parameters():
            p.requires_grad = False
        for name, p in self.llm.named_parameters():
            if ("wpe" in name) or ("ln_" in name) or ("ln_f" in name):
                p.requires_grad = True

    def forward(self, x):
        if x.dim() == 4:
            B, N, P, D = x.shape
            x = x.permute(0, 1, 3, 2).contiguous().view(B, N, D * P)  # (B, N, D*P)
        elif x.dim() == 3:
            B, N, _ = x.shape
        else:
            raise ValueError("TeacherNet expects input of shape (B, N, P, D) or (B, N, D*P)")

        x = self.linear_embedding(x)  # (B, N, n_embd)
        gpt_out = self.llm(inputs_embeds=x)
        gpt_last = gpt_out.last_hidden_state  # (B, N, n_embd)

        x_flat = gpt_last.reshape(B, -1)      # (B, N * n_embd)
        c = self.flatten_proj(x_flat)         # (B, output_dim)
        return c
