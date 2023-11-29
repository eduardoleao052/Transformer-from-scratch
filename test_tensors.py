import torch
from layers_torch import *
import torch.nn as nn
import torch.nn.functional as F

def test_tensor():
    k = torch.tensor([[[1,2,3,2,3,4],[4,5,6,5,6,7],[7,8,9,8,9,10],[10,11,12,11,12,13]],[[2,2,3,3,3,4],[4,5,7,5,6,7],[7,2,9,8,9,10],[1,2,12,2,3,13]]])


    k = k.view(2, 4, 3, 2).transpose(1,2)


    k = k.transpose(1,2).contiguous().view(2,4,6)
#===================================================# BACKPROP
    k = k.reshape(2,4,3,2).transpose(1,2)

    k = k.transpose(1,2).reshape(2,4,6)

    print(k)
    # print('==')
    # print(k)
    # print(q)
    # print(v)


def test_forward():
    torch.manual_seed(23)
    # (2, 4, 3)
    k = torch.tensor([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[2,2,3],[4,5,7],[7,2,9],[1,2,12]]],dtype=torch.float32)
    
    #dense = MultiHeadSelfAttention(3, 3, 3, 4)
    dense = TemporalDense(3,3)

    dense.set_mode('train')

    a = dense.forward(k)

    print(a)

    b = dense.backward(k)

    print(b)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, D, n_heads, n_timesteps, dropout, logger):
        super().__init__()
        self.key = nn.Linear(D, D, bias=False)
        self.query = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(n_timesteps,n_timesteps)))
        self.out_projection = nn.Linear(D, D, bias=False)
        self.att_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.H = D // n_heads # head_size
        self.logger = logger

        assert D % n_heads==0, "embedding dimension not divisible in equal heads."

    def forward(self,key):
        B, T, D = key.shape
        H = self.H

        k = self.key(key) # (B, T, D) @ (D, D) -> (B, T, D)
        q = self.query(key) # (B, T, D) @ (D, D) -> (B, T, D)
        v = self.value(key) # (B, T, D) @ (D, D) -> (B, T, D)

        k_heads = k.split(H, dim=-1) # num_heads * (B, T, H)
        q_heads = q.split(H, dim=-1) # num_heads * (B, T, H)
        v_heads = v.split(H, dim=-1) # num_heads * (B, T, H)

        k = torch.stack(k_heads, dim=1) # (B, num_heads, T, H)
        q = torch.stack(q_heads, dim=1) # (B, num_heads, T, H)
        v = torch.stack(v_heads, dim=1) # (B, num_heads, T, H)

        # (B, num_heads, T, H) @ (B, num_heads, H, T) -> (B, num_heads, T, T)
        att_activation = torch.einsum('bnTh, bnht -> bnTt',q, k.transpose(-2,-1)) 

        # Every row (0:T) in att[B, num_heads] keeps only first (T+1) words.
        att = att_activation.masked_fill(self.mask[:T,:T] == 0, float('-inf'))

        # Every row (0:T) in att[B, num_heads] becomes probability distribution of first (T+1) words.
        att = att/(H)**(0.5)
        logits = F.softmax(att, dim=-1)
        # logits = torch.exp(att - torch.max(att, axis=-1, keepdims=True)[0])
        # logits = logits / torch.sum(logits, axis= -1, keepdims=True) 

        logits = self.att_dropout(logits)

        # (B, num_heads, T, T) @ (B, num_heads, T, H) -> (B, num_heads, T, H)
        out = torch.einsum('bnTt, bnth -> bnTh', logits, v)

        out = out.transpose(1,2) # (B, num_heads, T, H) -> (B, T, num_heads, H)
        out = out.reshape(B,T,D) # (B, T, num_heads, H) -> (B,T,D)

        out = self.out_dropout(self.out_projection(out))
        return out


    def forward(self, x):
        # My implementation:
            # # (B, num_heads, T, H) @ (B, num_heads, H, T) -> (B, num_heads, T, T)
            # att_activation = torch.einsum('bnTh, bnht -> bnTt',q, k.transpose(-2,-1)) 

            # # Every row (0:T) in att[B, num_heads] keeps only first (T+1) words.
            # att = att_activation.masked_fill(self.mask[:T,:T] == 0, float('-inf'))

            # # Every row (0:T) in att[B, num_heads] becomes probability distribution of first (T+1) words.
            # att = att/(H)**(0.5)
            # logits = torch.exp(att - torch.max(att, axis=-1, keepdims=True)[0])
            # logits = logits / torch.sum(logits, axis= -1, keepdims=True) 

            # # (B, num_heads, T, T) @ (B, num_heads, T, H) -> (B, num_heads, T, H)
            # out = torch.einsum('bnTt, bnth -> bnTh', logits, v)

            # out = out.transpose(1,2) # (B, num_heads, T, H) -> (B, T, num_heads, H)
            # out = out.reshape(B,T,D) # (B, T, num_heads, H) -> (B,T,D)
            
        B, T, D = x.shape
        H = self.H
        num_heads = D // H

        k_stacked, q_stacked, v_stacked = self.att(x).split(D, dim=2) # (B, T, D) @ (D, 3 * D) -> 3 * (B, T, D)

        k = k_stacked.view(B, T, num_heads, H).transpose(1, 2) # (B, nh, T, H)
        q = q_stacked.view(B, T, num_heads, H).transpose(1, 2) # (B, nh, T, H)
        v = v_stacked.view(B, T, num_heads, H).transpose(1, 2) # (B, nh, T, H)

        att = (q @ k.transpose(-2, -1)) # (B, nh, T, H) @ (B, nh, H, T) -> (B, nh, T, T)

        # Reduces module sizes going into softmax:
        att = att / H**(.5)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        att = self.softmax(att, dim=-1)
        att = self.att_dropout(att)

        out = att @ v # (B, nh, T, T) @ (B, nh, T, H) -> (B, nh, T, H)
        
        # Restack heads in D dimension:
        out = out.transpose(1, 2).contiguous().view(B, T, D) 

        out = self.residual_proj(out) # (B, T, D) @ (D, D) -> (B, T, D)
        out = self.residual_dropout(out)

        self.cache = (att, k, v, q)

        return out


test_tensor()