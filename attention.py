import torch
from torch import nn
from torch.nn import functional as F
import math


class cross_attention(nn.Module):
    def __init__(
        self, num_heads, embed_dim, cross_dim, n_proj_bias=True, out_proj_bias=True
    ):
        super.__init__()
        self.in_proj1 = nn.Linear(cross_dim, embed_dim * 2, bias=n_proj_bias)
        self.in_proj2 = nn.Linear(embed_dim, embed_dim, bias=n_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x : latent (batch_size, seq_len, dim -number of features-) used for query
        # y :  prompt (batch_size, seq_len, dim -clip dim-) used for k and values - (batch, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        intermediate_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        # multiply q, k , v by the weights
        k, v = self.in_proj1(y).chunk(2, -1)
        q = self.in_proj2(x)

        # adding  for the number of heads dim and n features per head
        # transpose to be len_seq, d_head insteade of num_head, d_head
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        output = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        output = F.softmax(output, dim=-1) @ v

        output = output.transpose(1, 2).view(input_shape)

        return self.out_proj(output)


class self_attention(nn.Module):
    def __init__(
        self, num_heads: int, d_embedding: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        # define the wegiths w_q,w_k,w_v
        self.in_proj = nn.Linear(d_embedding, 3 * d_embedding, bias=in_proj_bias)
        # output weights
        self.out_proj = nn.Linear(d_embedding, d_embedding, bias=out_proj_bias)

        self.num_heads = num_heads
        self.d_head = d_embedding // num_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x : batch_size, sequence_length, d_embedding
        input_shape = x.shape
        batch_size, sequence_length, d_embedding = x.shape
        intermediate_shape = (batch_size, sequence_length, self.num_heads, self.d_head)

        # batch_size, sequence_length, 3*d_embedding -> batch_size, sequence_length, 3 * d_embedding ->
        # 3 * (batch_size, sequence_length, de_embedding)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # split the last dimension the embedding dimension into num_heads and d_head
        # batch_size, sequence_length, d_embedding -> batch_size, sequence_length, num_heads, d_head
        # batch_size, sequence_length, num_heads, d_head -> batch_size, num_heads, sequence_length, d_head
        q = q.view(*intermediate_shape).transpose(1, 2)
        k = k.view(*intermediate_shape).transpose(1, 2)
        v = v.view(*intermediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if causal_mask:
            # create a mask that will allow the model to only attend to the previous pixels
            mask = torch.ones_like(weight, dtype=bool).triu(1)
            weight.masked_fill(mask, -torch.inf)
        # soft max is applied on the last dimension making the sum of the weights equal to 1 in the row
        weight = F.softmax(weight, dim=-1)
        # apply the weights to the values
        # batch_size, num_heads, sequence_length, sequence_lenght @ batch_size, num_heads, sequence_length, d_head
        # giving final dimensions as -> batch_size, num_heads, sequence_length, d_head
        output = weight @ v
        # batch_size, num_heads, sequence_length, d_head -> batch_size, sequence_length, num_heads, d_head
        output = output.transpose(1, 2).reshape(input_shape)
        # batch_size, sequence_length, num_heads, d_head -> batch_size, sequence_length, d_embedding
        output = self.out_proj(output)

        return output
