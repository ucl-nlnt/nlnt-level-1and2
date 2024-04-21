from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
import torch

# step 1: load your model
bnb_config = BitsAndBytesConfig(            # BnB config is used to convert to 4 bit for lower VRAM
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained("model name", 
                                        device_map="auto",
                                        quantization_config=bnb_config,
                                        torch_dtype=torch.float16)

# step 2: load your tokenizer

tokenizer = AutoTokenizer.from_pretrained("model name")

# step 3: check for a pad token

keys = tokenizer.special_tokens_map.keys()
for i in keys: # check available special tokens
    print(i,':',tokenizer.special_tokens_map[i])

    # -> you need to check if there is an entry called 'pad_token'
    # -> if there is no pad_token, you need to add your own

# =========================================
# IGNORE THE REST OF THE STEPS DOWN HERE IF
# THE MODEL ALREADY HAS A PAD TOKEN
# =========================================

# step 4: add a pad token to your tokenizer

tokenizer.add_special_tokens({'pad_token':'[/PAD]'})
    # -> you can use any string as a pad token, just make sure that the sequence
    # will not appear in your dataset.

# step 5: adjust the embedding matrix in your model
model.resize_token_embeddings(len(tokenizer))

# step 6: make a directory where you will save this adjusted model and save
#   -> mkdir some_name

tokenizer.save_pretrained(save_directory="some_name")
model.save_pretrained(save_directory="some_name", save_embedding_layers=True)

# now once you saved this new modified model, you can use it as a model in the SFTTrainer
# after doing steps 1 to 6, you don't have to do it ever again for every time that you want to
# train the model

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
trainer.save_model('trained_model_directory')

