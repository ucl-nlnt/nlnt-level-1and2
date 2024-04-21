from dataclasses import dataclass, field
from typing import Optional
import random

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# modified from Dr. Atienza's training script

model_name = 'small_mixtral'
@dataclass
class ScriptArguments:

    model_name: Optional[str] = field(default=model_name, metadata={"help": "the model name"})
    #model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="training_data_post", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=5, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

    output_dir: Optional[str] = field(default=f"ft_small_mixtral", metadata={"help": "the output directory"})
    # output_dir: Optional[str] = field(default="finetune_output_synthetic_7B", metadata={"help": "the output directory"})
    # output_dir: Optional[str] = field(default="finetune_output_synthetic_mistral_7B", metadata={"help": "the output directory"})
    
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=4, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")

elif script_args.load_in_8bit or script_args.load_in_4bit:
    bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
    
else:
    device_map = None
    quantization_config = None
    torch_dtype = None


model = AutoModelForCausalLM.from_pretrained(script_args.model_name, 
                                        device_map="auto",
                                        quantization_config=bnb_config, # comment-out to train with full-precision
                                        torch_dtype=torch.float16)

print(model)
# Step 2: Load the dataset

dataset = load_dataset(script_args.dataset_name, split="train")

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=1,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    logging_dir="./logs",
    save_steps=4000,
    bf16=False,
    fp16=True,
    optim="adamw_torch",
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
        peft_config = LoraConfig(
        r=128,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
else:
    peft_config = None

def formatting_prompts_func(example):

    x = [i for i in example['text']]
    print(len(x))
    return x

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
#tokenizer.add_special_tokens({'pad_token':'[/PAD]'})
#tokenizer.save_pretrained("mistral_base")
tokenizer.padding_side = "right"
response_template = "|"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer = tokenizer)
model.resize_token_embeddings(len(tokenizer))
#model.save_pretrained(save_directory="mistral_base",save_embedding_layers=True)

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=580,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    #data_collator=collator
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)