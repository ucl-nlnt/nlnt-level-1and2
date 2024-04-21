from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import time

"""
bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
"""

cwd = "M4-ai/TinyMistral-6x248M"

# cwd = 'mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(cwd, padding_side="left")
tokenizer.padding_side = 'right'
for k in tokenizer.special_tokens_map.keys():
    print(k,':',tokenizer.special_tokens_map[k])

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cwd, device_map="auto")

# pad_token_id = tokenizer.encode(pad)[0]

def inference(model: AutoModelForCausalLM, nl_prompt:str, frame_num: int = 0):

    t_start = time.time()
    inputs = tokenizer([nl_prompt], truncation=True, add_special_tokens=True, return_tensors='pt', max_length=500).to('cuda')

    generate_ids = model.generate(**inputs, max_length=500)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
    delta_t = time.time() - t_start
    print(f"time taken: {round(delta_t,2)} | {round(len(tokenizer.tokenize(out)) / delta_t,2)} tokens per second.")
    return out

print(inference(model = model, nl_prompt='what are the top 2 largest countries in the world?', frame_num=1))