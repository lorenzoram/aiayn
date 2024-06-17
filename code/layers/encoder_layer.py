from layers.global_attention_layer import GlobalSelfAttention
from layers.feed_forward import FeedForward

import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = GlobalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
