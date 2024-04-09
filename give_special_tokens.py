import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login

notebook_login()

ckpt = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(ckpt)

print(tokenizer.special_tokens_map)
