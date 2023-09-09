# babyGPT

随手写的一个超级简单的GPT模型，可在上面对莎士比亚的文本进行预训练。

## Getting started

exps/pretrained包含一个在GeForce RTX 3090 24G显卡上训练的checkpoint，参数量为50M，训练时间小于1h，预训练语料为莎士比亚的文本，使用方法如下：
```bash
python pipeline.py --checkpoint_dir exps/pretrained --max_new_tokens 200 --prompt_str I
```
该checkpoint目录下还有一个使用checkpoint生成的样例文本文件output.txt

## TODO

- padding and attention masks
