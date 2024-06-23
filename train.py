from dataclasses import dataclass, field
from typing import Optional
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, AdaLoraConfig
from tqdm import tqdm
from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# modified from Dr. Atienza's training script


@dataclass
class ScriptArguments:

    #model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="training_data_post", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})

#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "microsoft/Phi-3-mini-128k-instruct"

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                        device_map="auto",
                                        quantization_config=bnb_config, # comment-out to train with full-precision
                                        torch_dtype=torch.float16, 
                                        trust_remote_code=True)
# Step 2: Load the dataset

print(model)

dataset = load_dataset(script_args.dataset_name, split="train")
eval_dataset = load_dataset(script_args.dataset_name, split="test")

# Step 3: Define the training arguments
output_dir = "phi3_force_test_Q4_e2_r256_new"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    logging_steps=10,
    num_train_epochs=2,
    max_steps=-1,
    report_to="wandb",
    logging_dir="~/logs",
    save_steps=454,
    bf16=False,
    fp16=True,
)

# Step 4: Define the LoraConfig


peft_config = LoraConfig(
    r=256,
    lora_alpha=512,
    # target_modules=["o_proj","q_proj","k_proj","v_proj","o_proj","gate_up_proj","down_proj"], # target module for Llama 3
    target_modules = ['o_proj',"qkv_proj","gate_up_proj","down_proj"], # target modules for Microsoft Phi 3
    task_type="CAUSAL_LM",
    use_rslora=True,
)

def formatting_prompts_func(example):

    x = [i for i in example['text']]
    print(len(x))
    return x

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=8000,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    #data_collator=collator
)

trainer.train()

# Step 6: Save the model
trainer.save_model(output_dir)
