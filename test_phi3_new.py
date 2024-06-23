from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, os
import uuid
from dataset_io import save_data_packet, open_data_packet
import progressbar
import json
import sys
import time, ast
import math

def generate_random_filename():

    random_filename = str(uuid.uuid4().hex)[:16]
    return random_filename

# checkpoint-11067  checkpoint-3689  checkpoint-7378
model_id = "phi3_latest_Q4_new_e2/checkpoint-7718"

bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config = bnb_config, # un-comment to quantize your model,
    trust_remote_code=True
)

errors = {}
entry_numbers = {}
maxes = {}

def compare_values(val1, val2, tolerance=0.1, key = None):
    """ Compare two values with a given tolerance for numeric types. """
    
    if isinstance(val1, float):
        
        if key not in errors.keys():
            errors[key] = 0.0
        errors[key] += abs(val1 - val2)
        if key not in entry_numbers.keys():
            entry_numbers[key] = 0.0
        entry_numbers[key] += 1
        if key not in maxes:
            maxes[key] = 0.0
        if maxes[key] < abs(val1 - val2):
            maxes[key] = abs(val1 - val2)
            
        return math.isclose(val1,val2,abs_tol=tolerance)
    
    elif isinstance(val1, str):
        return val1 == val2
    
    elif isinstance(val1,list):
        return val1 == val2
    
    return False

def inference(nl_prompt:str):

    number_of_input_tokens = len(tokenizer.tokenize(nl_prompt))
    t_start = time.time()
    inputs = tokenizer([nl_prompt], truncation=True, add_special_tokens=True, return_tensors='pt', max_length=2000).to('cuda')

    generate_ids = model.generate(**inputs, max_length=2500)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
    delta_t = time.time() - t_start
    print(f"time taken: {round(delta_t,2)} | {round((len(tokenizer.tokenize(out))-number_of_input_tokens)/ delta_t,2)} tokens per second.")
    return out

with open(os.path.join('training_data_post','eval','data.json'),'r') as f:
    prompts = json.loads(f.read())

x = 0
correct_labels = {"generation":0, "is possible":0, "### Possibility: True": 0, "### Possibility: False": 0, "Possibility generatation" : 0, "correct state" : 0}
wrong_labels = {"generation":0, "is possible":0, "### Possibility: True": 0, "### Possibility: False": 0, "Possibility generatation" : 0, "correct state" : 0}

for prompt in prompts:

    print('==========================')
    x += 1
    print("Progress:", round(x / len(prompts) * 100, 3), f"({x}/{len(prompts)})")

    if "The natural language prompt that I want you to break-down is:" in prompt:
        
        if "### Possibility: True" in prompt:

            inference_feed = prompt[:prompt.index("<|assistant|>")] + "<|assistant|>"
            prediction = inference(inference_feed)

            if "### Possibility" not in prediction:
                wrong_labels["Possibility generatation"] += 1
                continue
            else:
                correct_labels["Possibility generatation"] += 1

            if "### Possibility: True" not in prediction:
                wrong_labels["### Possibility: True"] += 1
            else:
                correct_labels["### Possibility: True"] += 1

        elif "### Possibility: False" in prompt:
            
            inference_feed = prompt[:prompt.index("<|assistant|>")] + "<|assistant|>"
            prediction = inference(inference_feed)

            if "### Possibility" not in prediction:
                wrong_labels["Possibility generatation"] += 1
                continue
            else:
                correct_labels["Possibility generatation"] += 1

            if "### Possibility: False" not in prediction:
                wrong_labels["### Possibility: False"] += 1
            else:
                correct_labels["### Possibility: False"] += 1
            

    elif """Given the following instruction breakdown, predict the next state. Use "<next_state_start>" and "<next_state_end>" to delineate your answer.""" in prompt:

        inference_feed = prompt[:prompt.rfind("{")]
        ground_truth = ast.literal_eval(prompt[prompt.rfind("{"):prompt.rfind("}") + 1])

        data_out = inference(inference_feed)
        try:
            predicted = ast.literal_eval(data_out[data_out.rfind("{"):data_out.rfind("}") + 1])
            
            correct = True
            for key in ground_truth.keys():

                if key not in correct_labels:
                    correct_labels[key] = 0
                    wrong_labels[key] = 0

                if key == 'execution length':
                    
                    if math.isclose(predicted[key],ground_truth[key],abs_tol=0.1): 
                        correct_labels[key] += 1
                    else: 
                        wrong_labels[key] += 1
                        correct = False
                
                else:

                    if predicted[key] == ground_truth[key]: 
                        correct_labels[key] += 1
                    else: 
                        wrong_labels[key] += 1
                        correct = False

            if correct:
                correct_labels['correct state'] += 1
            else:
                wrong_labels['correct state'] += 1

            correct_labels['generation'] = correct_labels['generation'] + 1

        except Exception as e:
            print(e)
            wrong_labels['generation'] = wrong_labels['generation'] + 1

    print("running stats:")
    for key in correct_labels.keys():
        
        if correct_labels[key] + wrong_labels[key] == 0:
            continue
        
        print(f"{key}:",round(correct_labels[key] / (correct_labels[key] + wrong_labels[key]) * 100,3), f" |              {correct_labels[key]} / {correct_labels[key] + wrong_labels[key]}")

print(correct_labels)
print(wrong_labels)