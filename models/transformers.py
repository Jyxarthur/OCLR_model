import math 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_

Tensor = torch.Tensor


class TransEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward):
        super().__init__()
        self.N = num_layers
        self.layers = get_clones(EncoderLayer(d_model, nhead, dim_feedforward), self.N)
    def forward(self, x, mask):
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x


class TransDecoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward):
        super().__init__()
        self.N = num_layers
        self.layers = get_clones(DecoderLayer(d_model, nhead, dim_feedforward), self.N)
    def forward(self, x, e_outputs, src_mask = None, trg_mask = None):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dim_feedforward, dropout = 0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout_1(self.self_attn(x,x,x,mask)))
        x = self.norm2(x + self.dropout_2(self.linear2(self.dropout(F.relu(self.linear1(x))))))
        return x
    
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.multihead_attn = MultiHeadAttention(heads, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x = self.norm1(x + self.dropout_1(self.self_attn(x, x, x, trg_mask)))
        x = self.norm2(x + self.dropout_2(self.multihead_attn(x, e_outputs, e_outputs,
        src_mask)))
        x = self.norm3(x + self.dropout_3(self.linear2(self.dropout(F.relu(self.linear1(x))))))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.in_proj_weight = Parameter(torch.empty(3 * d_model, d_model))
        xavier_uniform_(self.in_proj_weight)
        self.in_proj_bias = Parameter(torch.zeros(3 * d_model))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        k = k.view(bs, -1, self.h, self.d_k)
        q = q.view(bs, -1, self.h, self.d_k)
        v = v.view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out_proj(concat)
    
        return output
        
        
           
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores) 
    output = torch.matmul(scores, v)
    return output


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.weight = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.weight * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
   
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


