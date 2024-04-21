from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import time
import sys
import math
import ast
import random



cwd = "M4-ai/TinyMistral-6x248M"

# cwd = 'mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(cwd, padding_side="left")
print(tokenizer.special_tokens_map)
tokenizer.padding_side = 'left'

# Load evaluation dataset here.
eval_path = os.path.join(os.getcwd(),'training_data_post','eval','data.json')

with open(eval_path,'r') as f:
    data_json = json.loads(f.read())

list_of_eval_entries = []
for entry in data_json:

    text_prompt, answer = entry.split('### Answer:')
    text_prompt += '### Answer:'
    list_of_eval_entries.append([text_prompt, answer.split('</s>')[0], None])

random.shuffle(list_of_eval_entries)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cwd, device_map="auto")

# pad_token_id = tokenizer.encode(pad)[0]

def inference(model: AutoModelForCausalLM, prompts: list):

    t_start = time.time()
    inputs = tokenizer(prompts, padding=True, add_special_tokens=True, return_tensors='pt', max_length=600).to('cuda')

    generate_ids = model.generate(**inputs, max_length=600)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
    delta_t = time.time() - t_start
    print('Batch inference took', round(delta_t,2), 'seconds')
    print(out[0])
    return out


batch_size = 10
max_eval_size = 1000
q = min(max_eval_size,len(list_of_eval_entries))

state_number_t = 0
state_number_f = 0

orientation_t = 0
orientation_f = 0

dist_t = 0
dist_f = 0

exect_t = 0
exect_f = 0

twist_t = 0
twist_f = 0

complete_t = 0
complete_f = 0

frame_t = 0
frame_f = 0

keys = ['state num', 'orientation', 'distance', 'execution time', 'twist message', 'complete', 'correct frame']

for i in range(0,q,batch_size):

    print("Progress:", '(',i,'/',q,')',round(i / q,5) * 100)
    prompts = []

    for j in range(batch_size):
        prompts.append(list_of_eval_entries[i + j][0])
    
    inf = inference(model=model, prompts = prompts)

    #print('===================================')
    for j in range(batch_size):
        
        ground_truth = list_of_eval_entries[i + j][1].strip()
        prediction = inf[j].split('### Answer:')[1].split('</s>')[0].strip()
        list_of_eval_entries[i + j][2] = inf[j].split('### Answer:')[1].split('</s>')[0]
 
        # check model precisions
        gt = ast.literal_eval(ground_truth)
        pd = ast.literal_eval(prediction)

        correct_frame = True
        try:
            if gt['state number'] != pd['state number']: state_number_f += 1; correct_frame = False
            else: state_number_t += 1
        except:
            state_number_t += 1; correct_frame = False

        try:
            if math.isclose(gt['orientation'], pd['orientation'],rel_tol=0.025): orientation_t += 1
            else: orientation_f += 1; correct_frame = False
        except:
            orientation_f += 1; correct_frame = False

        try:    
            if math.isclose(gt['distance to next point'], pd['distance to next point'],rel_tol=0.025): dist_t += 1
            else: dist_f += 1; correct_frame = False
        except:
            dist_f += 1; correct_frame = False

        try:
            if math.isclose(gt['execution length'],pd['execution length'], rel_tol=0.025): exect_t += 1
            else: exect_f += 1; correct_frame = False
        except:
            exect_f += 1; correct_frame = False

        try:
            if gt['movement message'] != pd['movement message']: twist_f += 1; correct_frame = False
            else: twist_t += 1
        except:
            twist_f += 1; correct_frame = False

        try:    
            if gt['instruction complete'] != pd['instruction complete']: complete_f += 1; correct_frame = False
            else: complete_t += 1
        except:
            complete_f += 1; correct_frame = False

        if correct_frame: frame_t += 1
        else: frame_f += 1
    accuracy = [state_number_t / (state_number_t + state_number_f), orientation_t / (orientation_t + orientation_f),
                dist_t / (dist_t + dist_f), exect_t / (exect_t + exect_f), twist_t / (twist_f + twist_t),
                complete_t / (complete_f + complete_t), frame_t / (frame_t + frame_f)]
    
    accuracy = [round(i,4) * 100 for i in accuracy]

    print('===============================')
    print(f'CKPT: {24000}')
    for i in range(7):

        print(keys[i] + ' ' * (14 - len(keys[i])) + " :", accuracy[i])

print("Unrounded Accuracy:", accuracy)
with open(f'eval_test_ckpt{24000}_{q}_sm.json','w') as f:

    f.write(json.dumps(list_of_eval_entries[:q]))