import torch.nn as nn
import torch
import math


def positional_encoding(length, depth):
    angle_rads = torch.arange(length)[:, None] * torch.arange(depth)[None, :] / torch.pow(10000, (2 * (torch.arange(depth) // 2)) / depth)
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return angle_rads

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def forward(self, x):
        length = x.size(1)
        x = self.embedding(x)
        x *= math.sqrt(self.d_model)
        x = x + self.pos_encoding[:length, :].unsqueeze(0).to(x.device)
        return x
