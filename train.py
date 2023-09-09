"""train babyGPT on shakespeare"""
import os
import random
from tqdm import tqdm

import fire
import torch
from torch import optim

from model import babyGPTConfig, babyGPT, babyTokenizer


def sample_txt(s, l):
    idx = random.randint(0, len(s)-l)
    return s[idx:idx+l]

def train(
    steps: int = 1000,
    batch_size: int = 32,
    text_len: int = 256,
    lr: float = 3e-4,
    output_dir: str = 'exps/default',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('shakespeare.txt', 'r') as f:
        s = f.readlines()
    s = ''.join(s)
    train_s = s[:9*len(s)//10]
    valid_s = s[9*len(s)//10:]
    tokenizer = babyTokenizer()
    babyconfig = babyGPTConfig()
    babygpt = babyGPT(babyconfig)
    babygpt.to(device)
    optimizer = optim.AdamW(babygpt.parameters(), lr = lr)
    for step in tqdm(range(steps)):
        babygpt.train()
        train_text = [sample_txt(train_s, text_len) for _ in range(batch_size)]
        input_ids = torch.LongTensor(tokenizer.encode(train_text))
        input_ids = input_ids.to(device)
        _, loss = babygpt(input_ids, input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test output
        if step % 50 == 0:
            babygpt.eval()
            prompt_str = 'I'
            input_ids = torch.LongTensor(tokenizer.encode(prompt_str))
            input_ids = input_ids.to(device)
            out = babygpt.generate(input_ids, 100)
            print(tokenizer.decode(out)[0])

    babygpt.save_pretrained(output_dir)
    prompt_str = 'I'
    input_ids = torch.LongTensor(tokenizer.encode(prompt_str))
    input_ids = input_ids.to(device)
    babygpt.eval()
    out = babygpt.generate(input_ids, 100)
    print(tokenizer.decode(out)[0])

if __name__=='__main__':
    fire.Fire()
