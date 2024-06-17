import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        # Two linear layers with a ReLU activation in between
        # dropout is applied after the last linear layer
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate)
        )
        #We apply dropout [27] to the output of each sub-layer, before it is added to the
        #sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
        #positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
        #Pdrop = 0.1.
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.seq(x)
        x = x + residual
        x = self.layer_norm(x)
        return x
    