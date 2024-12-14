import torch
from torch import nn
from torch import functional as F
from attention import self_attention


class CLIP_embedding(nn.Module):
    def __init__(self, n_vocab: int, d_embedding: int, sequence_length: int):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, d_embedding)

        self.positional_embedding = nn.Parameter(
            torch.zeros(sequence_length, d_embedding)
        )

    def forward(self, tokens):
        # tokens : batch_size, sequence_length -> batch_size, sequence_length, d_embedding
        x = self.embedding(tokens)

        x = x + self.positional_embedding

        return x


class CLIP_layer(nn.Module):

    def __init__(self, num_heads: int, d_embedding: int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(d_embedding)
        self.attention = self_attention(num_heads, d_embedding)
        self.layernorm2 = nn.LayerNorm(d_embedding)
        self.linear_1 = nn.Linear(d_embedding, 4 * d_embedding)
        self.linear_2 = nn.Linear(4 * d_embedding, d_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x
        ###### Attention block ########
        x = self.layernorm1(x)
        x = self.attention(x, causal=True)
        x = self.layernorm2(x)
        x += residual
        ##### FEEDFORWARD #######
        residual = x
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.782 * x)  # Quick gelu function
        x = x.self.linear_2(x)
        x += residual
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding the text -> 49408 vocabulary size , 768 features(d), 77 tokens max length
        self.embedding = CLIP_embedding(49408, 768, 77)
        # 12 layers of multihead attention each has 12 heads
        self.layers = nn.ModuleList([CLIP_layer(12, 768) for i in range(12)])

        # layer normalization
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # toeken (batch_size, sequence_length) -> (batch_size, sequence_length, d_embedding)
        tokens = tokens.dtype(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        # batch_size,sequence_length, d_embedding
        output = self.layer_norm(state)

        return output
