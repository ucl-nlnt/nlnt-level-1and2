from dataclasses import dataclass, field
from typing import Optional
import random

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

# loaded pretrained

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.special_tokens_map)

tokenizer.add_special_tokens({'pad_token':'[/PAD]'})
tokenizer.save_pretrained(save_directory="llama3_q4")
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                        device_map="auto",
                                        quantization_config=bnb_config,
                                        torch_dtype=torch.float16)

model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(save_directory="llama3_q4", save_embedding_layers=True)

