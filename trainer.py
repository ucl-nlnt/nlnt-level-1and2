import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login

notebook_login()

ckpt = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt)

print(tokenizer.special_tokens_map)
