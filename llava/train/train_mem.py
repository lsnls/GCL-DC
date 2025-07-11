# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import os

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "7741ab02c102174d64241382f036a78587a9f145"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "qllava-wsi"
    train()
