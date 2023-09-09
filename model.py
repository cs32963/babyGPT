"""babyGPT"""
import os
import json
import string
import random
from tqdm import tqdm
from typing import List, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

class babyTokenizer:
    def __init__(self, max_len = 256):
        self.vocab = string.printable
        self.vocab_size = len(self.vocab)
        self.vocab2id = {c:i for i, c in enumerate(self.vocab)}
        self.id2vocab = {i:c for i, c in enumerate(self.vocab)}
        self.max_len = max_len

    def encode(self, s: Union[str, List[str]]):
        s = [s] if isinstance(s, str) else s
        s = [_[:self.max_len] for _ in s]
        return [list(map(lambda x: self.vocab2id[x], _)) for _ in s]

    def decode(self, ids: List[Union[int, List[int]]]):
        ids = [ids] if isinstance(ids[0], int) else ids
        return [''.join(map(lambda x: self.id2vocab[x], _)) for _ in ids]

@dataclass
class babyGPTConfig:
    vocab_size: int = len(string.printable)
    hidden_size: int = 32
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.5
    max_len: int = 256

    def load(self, path):
        with open(path, 'r') as f:
            loaded_dict = json.load(f)
        for k, v in loaded_dict.items():
            setattr(self, k, v)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

class babyAttention(nn.Module):
    def __init__(self, config: babyGPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer('causal_mask', torch.tril(torch.ones((config.max_len, config.max_len))))

    def forward(self, x):
        B, T, C = x.shape
        # B * self.num_heads * T * self.head_dim
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # B * self.num_heads * T * self.head_dim
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # B * self.num_heads * T * T
        atten_weights = q @ k.transpose(-2, -1) * self.head_dim**(-0.5)
        atten_weights = atten_weights.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        atten_weights = F.softmax(atten_weights, dim=-1)

        # B * self.num_head * T * self.head_dim
        o = atten_weights @ v
        # B * T * self.num_head * self.head_dim
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        # B * T * C
        o = self.o_proj(o)
        return o
        
class babyMLP(nn.Module):
    def __init__(self, config: babyGPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.ReLU(),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class babyLayer(nn.Module):
    def __init__(self, config: babyGPTConfig):
        super().__init__()
        self.config = config
        self.attention = babyAttention(config)
        self.att_ln = nn.LayerNorm(config.hidden_size)
        self.mlp = babyMLP(config)
        self.mlp_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = x + self.attention(self.att_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x

class babyGPT(nn.Module):
    def __init__(self, config: babyGPTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_len, config.hidden_size)
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.layers = nn.ModuleList([babyLayer(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.register_buffer('pos_ids', torch.arange(0, self.max_len))

    def forward(self, x, labels=None):
        # B * T * C
        B, T = x.shape
        vocab_embed = self.embedding(x)
        pos_embed = self.pos_embedding(self.pos_ids[:T])
        x = vocab_embed + pos_embed
        for layer in self.layers:
            x = layer(x)
        # B * T * vocab_size
        logits = self.lm_head(x)
        if labels is not None:
            loss_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
            loss_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(loss_logits, loss_labels)
            return logits, loss
        else:
            return logits

    @torch.no_grad()
    def generate(self, x, max_new_token = 64, sampling=True):
        # x has shape B * T
        for i in tqdm(range(max_new_token)):
            context = x[:, -self.max_len+1:]
            # B * vocab_size
            logits = self.forward(context)[:, -1, :]
            if sampling:
                prob = logits.softmax(dim=-1)
                new_ids = torch.multinomial(prob, 1)
            else:
                new_ids = logits.argmax(dim=-1, keepdim=True)
            x = torch.cat((x, new_ids), dim=1)
        return x.tolist()
    
    def save_pretrained(self, save_dir: str):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=4, ensure_ascii=False)
        # save model state dict
        torch.save(self.state_dict(), os.path.join(save_dir, 'checkpoint.pt'))

if __name__=='__main__':
    babyconfig = babyGPTConfig()
    babygpt = babyGPT(babyconfig)
    x = [[random.randint(0, len(string.printable)-1) for _ in range(5)] for _ in range(3)]
    x = torch.LongTensor(x)
    logits, loss = babygpt(x, x)
    breakpoint()
    print()
