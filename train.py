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



@dataclass
class ScriptArguments:

    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="/home/gabriel/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55", metadata={"help": "the model name"})
    # model_name: Optional[str] = field(default="/home/gabriel/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="training_data_post", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-2, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=2900, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

    output_dir: Optional[str] = field(default="ft_mistral_7B", metadata={"help": "the output directory"})
    # output_dir: Optional[str] = field(default="finetune_output_synthetic_7B", metadata={"help": "the output directory"})
    # output_dir: Optional[str] = field(default="finetune_output_synthetic_mistral_7B", metadata={"help": "the output directory"})
    
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")

elif script_args.load_in_8bit or script_args.load_in_4bit:
    bnb_config = BitsAndBytesConfig(
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
                                         quantization_config=bnb_config,
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
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    logging_dir="./logs",
    save_steps=2500,
    bf16=False,
    fp16=True
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
        peft_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
else:
    peft_config = None

def formatting_prompts_func(example): # Implemented by Kaquilala

    x = [i for i in example['text']]
    print(len(x))
    return x
    
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side = "right")
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)