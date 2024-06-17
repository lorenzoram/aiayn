#We also modify the self-attention
#sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
#masking, combined with fact that the output embeddings are offset by one position, ensures that the
#predictions for position i can depend only on the known outputs at positions less than i

import torch.nn as nn
import torch
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        # I was kinda confused here so I had to look this up (Lorenzo)
        # https://discuss.pytorch.org/t/attn-mask-in-nn-multiheadattention/173603/3
        # Source: Theo
        #FIXME: check if this is what we want to do
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)  # Create causal mask
        attn_output, _ = self.mha(query=x, key=x, value=x, attn_mask=attn_mask)
        x = x + attn_output
        x = self.layer_norm(x)
        return x
