import torch.nn as nn
import torch


class GlobalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(GlobalSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.mha(query=x, key=x, value=x)
        x = x + attn_output
        x = self.layer_norm(x)
        return x
