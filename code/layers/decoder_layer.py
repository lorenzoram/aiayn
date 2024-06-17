import torch.nn as nn
import torch
from layers.masked_self_attention_layer import CausalSelfAttention
from layers.global_attention_layer import GlobalSelfAttention
from layers.feed_forward import FeedForward
from layers.attention_layer import AttentionLayer

class DecoderLayer(nn.Module):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.cross_attention = AttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def forward(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)

        x = self.ffn(x)
        return x
    