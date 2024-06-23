import os
import zlib
import json
import quart_funcs
import random
# from nlnt_types import StatesList
from transformers import AutoTokenizer
import math
import sys
from collections import deque
import shutil

precision = 3

def open_data_packet(path):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

def quaternion_to_yaw(x, y, z, w):  # Generated by GPT-4
    """
    Convert a quaternion into yaw (rotation around z-axis in radians)
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def find_keyframes_indexes(state_packet):

    frames = state_packet['states']


    #for i in range(len(frames)):

    twist_last = frames[0]['twist']
    keyframes = [0]
    #print(twist_last['linear'], twist_last['angular'])

    for i, frame in enumerate(frames):

        if frame['twist']['linear'] != twist_last['linear'] or frame['twist']['angular'] != twist_last['angular']:
            #print(frame['twist']['linear'], twist_last['linear'], frame['twist']['angular'], twist_last['angular'])
            twist_last = frame['twist']
            keyframes.append(i)

    return keyframes

def prep_frame_data(state_list: list, keyframe_indexes):

    data = [state_list[i] for i in keyframe_indexes]

    # shift and rotate odometry data
    reference_position = data[0]['odometry']['pose_position']
    reference_rotation = data[0]['odometry']['pose_orientation_quarternion']

    inverse_reference_rotation = quart_funcs.inverse_quarternion(
        reference_rotation)

    for i, frame in enumerate(data):

        odom = frame['odometry']
        old_position, old_rotation = odom['pose_position'], odom['pose_orientation_quarternion']

        new_position = quart_funcs.adjust_position_origin(
            reference_position, old_position)
        new_rotation = quart_funcs.adjust_orientation_origin(
            inverse_reference_rotation, old_rotation)

        odom['pose_position'] = new_position
        odom['pose_orientation_quarternion'] = new_rotation

        data[i]['odometry'] = odom

    return data # odometry data

def compute_distance(coord1, coord2):

    total = 0
    for i in range(3):
        total += (coord2[i] - coord1[i])**2

    return math.sqrt(total)

def phi3_format(chat_history: list):

    output = ""
    for i in chat_history:
        
        if i['role'] == 'user':
            output += '<|user|>' + '\n'

        elif i['role'] == 'assistant' or i['role'] == 'assist':
            output += '<|assistant|>' + '\n'

        output += i['content'].strip() + '<|end|>' + '\n'

    output = output.strip()

    return output

def split_into_train_and_eval(some_list: list):

    # uses 80-20 split
    random.shuffle(some_list)
    train_max = int(len(some_list) * 0.8)
    train = some_list[:train_max]
    eval_set = some_list[train_max:]

    return (train, eval_set)
    
training_entries_impossible = []
# open impossible and rephrased

import ast


list_of_paths = [os.path.join('training_data_pre_rephrased',i) for i in os.listdir('training_data_pre_rephrased')]
files = {}

if not os.path.exists('file.json'):
    x = 0
    for path in list_of_paths:

        print(f'extracting labels; {round(x / len(list_of_paths) * 100, 3)}')
        q = open_data_packet(path)
        files[q['natural_language_prompt'].strip()] = path
        x += 1

    with open('file.json','w') as f:
        f.write(json.dumps(files))

else:
    
    with open('file.json','r') as f:
        files = json.loads(f.read())

eval_set = os.path.join(
    'training_data_post',
    'eval',
    'data.json'
)

with open(eval_set, 'r') as f:

    list_of_data = json.loads(f.read())

print(list_of_data[0])

eval_split_possible_paths = []
return_arr = []

skipped = 0
for entry in list_of_data:

    if "I need you to break-down this natural language prompt into a series of steps" in entry: # breakdown phase
        return_arr.append(entry)
        continue

    elif "Given the following instruction breakdown, predict the next state" in entry: # next-state prediction"

        if '<prompt_start>' in entry:

            prompt = entry[entry.index("<prompt_start>") + len("<prompt_start>"):entry.index("<prompt_end>")]
            prompt = prompt.strip()

        else:

            prompt = entry[entry.index("<prompt>") + len("<prompt>"):entry.index("<prompt_end>")]
            prompt = prompt.strip()

        try:

            prompt = prompt.strip()
    
            if files[prompt]not in eval_split_possible_paths: eval_split_possible_paths.append(files[prompt])
            
        except Exception as e:
            skipped += 1
            print(f"Skipped {prompt}")

print('skipped:', skipped)
# handle actual possible prompts here

eval_examples_breakdowns = []
eval_examples_state_predictions = []

possible_skipped = 0
progress_possible = 1

print(len(eval_split_possible_paths))
split1 = []
split2 = []
split3 = []
split4 = []
impossibles = []

asd = 1
for iter, i in enumerate(eval_split_possible_paths):
    break
    print('processing:', iter, '/', len(eval_split_possible_paths))
    print(f"Progess:{progress_possible / len(eval_split_possible_paths) * 100}")
    data = open_data_packet(i)
    
    progress_possible += 1
    if "### Possibility: False" in data['explanation']: 
        print(data['explanation'])
        possible_skipped += 1
        continue

    # handle states here
    state_history = []
    keyframes = find_keyframes_indexes(data)
    prepped = prep_frame_data(data['states'], keyframes) # list of all keyframes

    skipped_frames = 0

    kstates_pruned = []
    for i in prepped:
        
        if i['twist']['linear'][0] == 0.0 and i['twist']['angular'][2] == 0.0:
            continue
        
        kstates_pruned.append(i)

    #print(data['natural_language_prompt'])
    for i, data_frame in enumerate(prepped):

        text = data['explanation'].replace("<explanation_start>","")
        
        current_frame = prepped[i - 1]
        next_frame = prepped[i]

        if [current_frame['twist']['linear'][0], current_frame['twist']['angular'][2]] == [0.0, 0.0]:
            
            skipped_frames += 1
            continue

        #print(hex(i - skipped_frames), hex(i - 1), i)
        twist_mess = [current_frame['twist']['linear'][0], current_frame['twist']['angular'][2]]
        instruction = "stop"

        if twist_mess[0] != 0.0:
            instruction = "move forwards"

        elif twist_mess[0] == 0.0 and twist_mess[1] > 0.0:
            instruction = "turn left"
        
        elif twist_mess[0] == 0.0 and twist_mess[1] < 0.0:
            instruction = "turn right"
        
        execution_length = round(next_frame['twist']['time'] - current_frame['twist']['time'], precision)
        if execution_length == 0.0:
            skipped_frames += 1
            continue

        instruction_str = {
            "total states" : len(kstates_pruned) - 1,
            "state number" : i - skipped_frames,
            "action" : instruction,
            "twist message" : [current_frame['twist']['linear'][0], current_frame['twist']['angular'][2]],
            "execution length" : round(next_frame['twist']['time'] - current_frame['twist']['time'], precision)
        }

        #print(instruction_str)
        instruction_str = json.dumps(instruction_str)
        
        if state_history != []:
            string_prompt = f"""
Given the following instruction breakdown, predict the next state. Use "<next_state_start>" and "<next_state_end>" to delineate your answer.

<prompt> {data['natural_language_prompt']} <prompt_end>
<state_history> {state_history} <state_history_end>

<breakdown>
{text}
<breakdown_end>
""".strip()
        else:
            string_prompt = f"""
Given the following instruction breakdown, predict the next state. Use "<next_state_start>" and "<next_state_end>" to delineate your answer.

<prompt_start> {data['natural_language_prompt']} <prompt_end>
<state_history_start> [None] <state_history_end>

<breakdown>
{text}
<breakdown_end>
""".strip()

        model_answer = f"""
The next state is:
<next_state_start>
{instruction_str}
<next_state_end>
""".strip()

        chat = [
        {"role":"user","content" : string_prompt},
        {"role":"assist", "content" : model_answer}]

        if asd == 1:
            split1.append(phi3_format(chat))

        elif asd == 2:
            split2.append(phi3_format(chat))

        elif asd == 3:
            split3.append(phi3_format(chat))

        elif asd == 4:
            split4.append(phi3_format(chat))
        
        # print(action_examples[-1])
        #print(training_examples_breakdowns[-1])
        state_history.append(ast.literal_eval(instruction_str))
        if len(state_history) >= 6:
            state_history = state_history[1:]

    asd += 1
    if asd == 5:
        asd = 0

asd = 1

for entry in return_arr:

    data = entry[entry.rfind('<explanation_start>'):entry.rfind('<explanation_end>')]
    if '### Possibility: False' in data:
        print('============================')
        print(entry)
        impossibles.append(entry)

    if asd == 1:
        split1.append(entry)

    elif asd == 2:
        split2.append(entry)

    elif asd == 3:
        split3.append(entry)

    elif asd == 4:
        split4.append(entry)

    asd += 1
    if asd == 5:
        asd = 0

out = [split1, split2, split3, split4, impossibles]
with open('revised.json', 'w') as f:
    f.write(json.dumps(out))