import math
import numpy as np
from ..core.tensor import Tensor
from ..core.nn import nn, F


class LAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        use_relative_positions=False,
        max_seq_len=512,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_relative_positions = use_relative_positions
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

        if self.use_relative_positions:
            self.relative_position_embeddings = Tensor.randn(
                2 * max_seq_len - 1, self.head_dim
            )

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.data.shape
        seq_len_k = key.data.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_reshaped = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        q = q_reshaped.transpose(1, 2)
        
        k_reshaped = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        k = k_reshaped.transpose(1, 2)
        
        v_reshaped = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v_reshaped.transpose(1, 2)

        attn_scores = Tensor.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.use_relative_positions:
            relative_positions = (
                np.arange(seq_len_q)[:, None] - np.arange(seq_len_k)[None, :]
            )
            relative_positions = relative_positions + self.max_seq_len - 1
            relative_positions = np.clip(
                relative_positions, 0, 2 * self.max_seq_len - 2
            )

            relative_embeddings = self.relative_position_embeddings.data[
                relative_positions
            ]

            q_reshaped = q.data.reshape(
                batch_size * self.num_heads, seq_len_q, self.head_dim
            )
            relative_attn = np.matmul(q_reshaped, relative_embeddings.transpose(1, 2))
            relative_attn = relative_attn.reshape(
                batch_size, self.num_heads, seq_len_q, seq_len_k
            )
            attn_scores = attn_scores + Tensor(relative_attn) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float("inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        output = Tensor.matmul(attn_weights, v)

        output = output.transpose(1, 2).view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights

    def get_config(self):
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "bias": self.q_proj.bias is not None,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=config.get("dropout", 0.1),
            bias=config.get("bias", True),
        )

    def save_config(self, path):
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, ensure_ascii=False, indent=2)

    def load_config(self, path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return self.from_config(config)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.attention = LAttention(embed_dim, num_heads, dropout, bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        output, _ = self.attention(query, key, value, attn_mask, key_padding_mask)
        return output


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.attention = LAttention(embed_dim, num_heads, dropout, bias)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        output, _ = self.attention(x, x, x, attn_mask, key_padding_mask)
        return output


class SparseAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, top_k=32, dropout=0.1, bias=True):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.data.shape
        seq_len_k = key.data.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = Tensor.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float("inf"))

        top_k = min(self.top_k, seq_len_k)
        top_attn_scores, top_indices = attn_scores.topk(top_k, dim=-1)

        attn_weights = F.softmax(top_attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        top_indices_reshaped = top_indices.data.reshape(
            batch_size, self.num_heads, seq_len_q, top_k
        )

        batch_indices = np.arange(batch_size)[:, None, None, None]
        head_indices = np.arange(self.num_heads)[None, :, None, None]

        selected_v = v.data[batch_indices, head_indices, top_indices_reshaped, :]

        attn_weights_reshaped = attn_weights.data.reshape(
            batch_size, self.num_heads, seq_len_q, top_k, 1
        )
        weighted_sum = np.sum(selected_v * attn_weights_reshaped, axis=-2)

        output = (
            Tensor(weighted_sum)
            .transpose(1, 2)
            .view(batch_size, seq_len_q, self.embed_dim)
        )
        output = self.out_proj(output)

        return output, attn_weights


class LinearAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.data.shape
        seq_len_k = key.data.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        q = F.relu(q)
        k = F.relu(k)

        kv = Tensor.matmul(k.transpose(-2, -1), v)
        qkv = Tensor.matmul(q, kv)

        q_norm = q.sum(dim=-1, keepdim=True)
        k_norm = k.sum(dim=-2, keepdim=True)
        norm = Tensor.matmul(q_norm, k_norm)
        qkv = qkv / (norm + 1e-8)

        output = qkv.transpose(1, 2).view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)

        return output
