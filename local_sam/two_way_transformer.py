import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation=nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim,
                    num_heads,
                    mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, queries, keys, query_pos, key_pos):
        q = queries
        k = keys
        for layer in self.layers:
            q, k = layer(q, k, query_pos, key_pos)
        q = q + self.final_attn_token_to_image(
            q + query_pos, k + key_pos
        )
        q = self.norm_final_attn(q)
        return q, k


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation=nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.self_attn = Attention(
            embedding_dim, num_heads
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

    def forward(self, queries, keys, query_pos, key_pos):
        q = queries + self.self_attn(self.norm1(queries))
        q = q + self.cross_attn_token_to_image(
            self.norm2(q + query_pos), keys + key_pos
        )
        q = q + self.mlp(self.norm3(q))

        k = keys + self.cross_attn_image_to_token(
            self.norm4(keys + key_pos), q + query_pos
        )
        return q, k


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def forward(self, q, k):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(k)

        attn = torch.einsum("bqd,bkd->bqk", q, k)
        attn = attn / (q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bqk,bkd->bqd", attn, v)
        out = self.out_proj(out)
        return out


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, activation=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.activation = activation()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
