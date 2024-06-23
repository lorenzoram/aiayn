import torch
import torch.nn as nn
import math
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Paper: In the embedding layers, we multiply those weights by sqrt(d_model).
        # TODO: check if we can use torch here
        # return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        return self.embedding(x) * math.sqrt(self.d_model)
    
# Positional encoding explained, vector of size d_model that contains information about the position of the token in the sequence.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # (Seq_len, 1), pos
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Denominator in logspace for numerical stability, slightly off but better according to video
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Sin for even indices, cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Keep in model, but not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# Rember we use this straight from torch, # TODO: check
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(1))
        self.bias = nn.Parameter(torch.tensor(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    # Paper: linear layer with 2 matrices and relu in between
    # d_model = 512, dff = 2048
    def __init__(self, d_model: int, dff: int, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model) --> (batch_size, seq_len, dff) --> (batch_size, seq_len, d_model)
        return self.seq(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # If we don't want to words to interact, we can use a mask, close to zero -> -inf
        # Hide attention for 2 words
        query = self.w_q(q) # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, d_model)

        # Divide into h heads, we split the embedding into h heads
        
        # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, h, d_k) ->> (batch_size, h, seq_len, d_k)
        # Each head should see the full sentence, but only a part of the embedding ...
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (batch_size, seq_len, d_model) ->> (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Calculate attention

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) ->> (batch_size, seq_len, h, d_k) ->> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, d_model)
        return self.w_o(x)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer): # sublayer is the previous layer
        # We first apply the sublayer, 
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    
    # src_mask is a mask that hides the padding
    def forward(self, x, src_mask):
        # Role of query and key is the same, value is the input, each word can attend to all the words
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        # 3 residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, enc_output, src_mask, trg_mask):
        # Translation task: second mask from decoder (trg_mask), first mask from encoder (src_mask)
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, enc_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        return self.norm(x)
    
# Converts from d_model to vocab_size, basically a linear layer that converts the output to the vocab size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (batch_size, seq_len, d_model) ->> (batch_size, seq_len, vocab_size)
        # Uses log softmax for numerical stability
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, trg_embedding: InputEmbedding,
                  src_pos: PositionalEncoding, trg_pos: PositionalEncoding, projection: ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection = projection
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, enc_output, src_mask, trg, trg_mask):
        trg = self.trg_embedding(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, enc_output, src_mask, trg_mask)
    
    def project(self, x):
        return self.projection(x)
    

def build_transformer(src_vocab_len: int, trg_vocab_len: int, src_seq_len: int, trg_seq_len: int, d_model: int = 512, n:int = 6, h:int =8,
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Embeddings first
    src_embedding = InputEmbedding(d_model, src_vocab_len)
    trg_embedding = InputEmbedding(d_model, trg_vocab_len)

    # Positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # n encoder layers
    encoder_blocks = nn.ModuleList([EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(n)])

    # n decoder layers
    decoder_blocks = nn.ModuleList([DecoderBlock(MultiHeadAttention(d_model, h, dropout),
                                                  MultiHeadAttention(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(n)])

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    # Final projection layer

    projection = ProjectionLayer(d_model, trg_vocab_len)

    transformer = Transformer(encoder, decoder, src_embedding, trg_embedding, src_pos, trg_pos, projection)    

    # Parameters initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer