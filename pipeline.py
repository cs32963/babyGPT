"""pipeline"""
import os

import fire
import torch
import humanize

from model import babyGPTConfig, babyGPT, babyTokenizer

def main(
    prompt_str: str = 'I',
    max_new_tokens: int = 100,
    sampling: bool = True,
    checkpoint_dir: str = '',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = babyTokenizer()
    babyconfig = babyGPTConfig()

    if checkpoint_dir:
        babyconfig.load(os.path.join(checkpoint_dir, 'config.json'))
        babygpt = babyGPT(babyconfig)
        babygpt.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), map_location=device))
    else:
        babygpt = babyGPT(babyconfig)

    babygpt.to(device)
    babygpt.eval()
    num_params = humanize.naturalsize(sum(p.numel() for p in babygpt.parameters()))
    print(f'number of parameters: {num_params}')

    input_ids = torch.LongTensor(tokenizer.encode(prompt_str))
    input_ids = input_ids.to(device)
    out = babygpt.generate(input_ids, max_new_tokens, sampling)
    print(tokenizer.decode(out)[0])

if __name__=='__main__':
    fire.Fire(main)
