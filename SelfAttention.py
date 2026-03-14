import torch

class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        torch.manual_seed(123)
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        query = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_score = query @ keys.T

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        return attn_weights @ values

class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        torch.manual_seed(123)
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_score = queries @ keys.T

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_score / d_k ** 0.5, dim=-1)
        return attn_weights @ values
class CasualAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
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
class MultiHeadAttensionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_head, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_head)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)