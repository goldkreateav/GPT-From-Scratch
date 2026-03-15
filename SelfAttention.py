import torch
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        query = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_score = query @ keys.T

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        return attn_weights @ values

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_score = queries @ keys.T

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        return attn_weights @ values

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_score = queries @ keys.transpose(1, 2)

        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values

class MultiHeadAttensionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_head, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_head)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttension(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_out // num_heads
        self.out_proj = nn.Linear(d_out, d_out)
        self.num_heads = num_heads
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_score = queries @ keys.transpose(2, 3)

        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec