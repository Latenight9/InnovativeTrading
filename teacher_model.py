import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class TeacherNet(nn.Module):
    def __init__(self, input_dim=2, patch_size=6, embedding_dim=768, output_dim=128):
        super(TeacherNet, self).__init__()

        # ⬛ 1. Lineares Embedding: Patch → LLM-kompatibles Token
        self.linear_embedding = nn.Linear(patch_size * input_dim, embedding_dim)

        # ⬛ 2. GPT2-Konfiguration mit 6 Layern
        config = GPT2Config(
            n_embd=embedding_dim,
            n_layer=6,
            n_head=8,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )
        self.gpt2 = GPT2Model(config)

        # ⬛ 3. GPT2 einfrieren – nur LayerNorm & PosEmb trainierbar
        for name, param in self.gpt2.named_parameters():
            if "attn" in name or "mlp" in name:
                param.requires_grad = False
            elif "ln" in name or "wpe" in name:
                param.requires_grad = True

        # ⬛ 4. Flatten + Linear → Ausgabe-Vektor c
        self.flatten_proj = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor (batch, n_patches, patch_size, input_dim)
        Gibt Feature-Vektor c zurück: Tensor (batch, output_dim)
        """
        batch_size, n_patches, patch_size, input_dim = x.shape

        # ⬛ Flatten jeden Patch zu einem Vektor
        x = x.reshape(batch_size, n_patches, patch_size * input_dim)

        # ⬛ Lineares Embedding
        x = self.linear_embedding(x)  # → (batch, n_patches, embedding_dim)

        # ⬛ GPT2-Verarbeitung (inkl. Positional Embedding)
        outputs = self.gpt2(inputs_embeds=x)
        last_hidden = outputs.last_hidden_state  # → (batch, n_patches, embedding_dim)

        # ⬛ Nehme letztes Token als Repräsentation & projiziere es
        z = last_hidden[:, -1, :]  # alternativ: torch.mean(last_hidden, dim=1)
        c = self.flatten_proj(z)   # → (batch, output_dim)

        return c
