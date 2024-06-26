from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import zlib
import json
import quart_funcs
import random
from transformers import AutoTokenizer
import math
from collections import deque
import time

def get_list_of_files(folder):

    p = os.getcwd()
    folder_path = os.path.join(p, folder)
    return [os.path.join(folder_path, i) for i in os.listdir(folder_path)]

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

    twist_last = frames[0]['twist']
    keyframes = [0]
    for i, frame in enumerate(frames):
        if frame['twist'] != twist_last:
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

    return data

def compute_distance(coord1, coord2):

    total = 0
    for i in range(3):
        total += (coord2[i] - coord1[i])**2

    return math.sqrt(total)

def create_training_entry_from_packet(data_packet, keyframe_indexes):

    # Gemma: {'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
    # Mistral: // to be filled out later
    # Llama: // to be filled out later

    prompt = data_packet['natural_language_prompt']
    kstates = prep_frame_data(data_packet['states'], keyframe_indexes)
    training_examples = []
    
    entry_template = f"""You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: {prompt}."""

    frame_history_list = deque([])

    precision = 3
    skipped_frames = 0

    for i in range(1, len(kstates)):
        
        current_frame = kstates[i-1]
        next_frame = kstates[i]
        try:

            twist_instruction = (current_frame['twist']['linear'][0], current_frame['twist']['angular'][2])

            if twist_instruction == (0.0, 0.0):
                #print('skipped stop')
                skipped_frames += 1
                continue

            frame = {

                "state number" : hex(i - 1 - skipped_frames),
                "orientation" : round(quaternion_to_yaw(*current_frame['odometry']['pose_orientation_quarternion']), precision),
                "distance to next point" : round(abs(compute_distance(current_frame['odometry']['pose_position'], next_frame['odometry']['pose_position'])),precision),
                "execution length" :  round(next_frame['twist']['time'] - current_frame['twist']['time'], precision),
                "movement message" : (current_frame['twist']['linear'][0], current_frame['twist']['angular'][2]),
                "instruction complete" : "#ongoing" if i != len(kstates)-1 else '#complete'

            }

        except KeyError:
            print('Timeless twist message detected.')
            return [-1] # will be eliminated later

        if i == 1: # first iteration.
            entry = entry_template + ' | ' + 'History: [ None ] ' + "### Answer: " + str(frame) + ' </s>'
            frame_history_list.append(frame)
            training_examples.append(entry)
            continue
        
        else:

            entry = entry_template + ' | ' + f'History: {list(frame_history_list)} ' + "### Answer: " + str(frame) + ' </s>'
            training_examples.append(entry)

        frame_history_list.append(frame)
        if len(frame_history_list) > 5: frame_history_list.popleft() # maximum history length
        
    return training_examples

def create_eval_pair(task_str: str):

    prompt, answer = task_str.split('### Answer:')
    prompt = prompt.strip() + ' ### Answer:'

    return (prompt, answer)


def inference(model: AutoModelForCausalLM, prompts: list):

    t_start = time.time()
    inputs = tokenizer(prompts, padding=True, add_special_tokens=True, return_tensors='pt', max_length=600).to('cuda')

    generate_ids = model.generate(**inputs, max_length=600)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
    delta_t = time.time() - t_start
    print('Batch inference took', round(delta_t,2), 'seconds')
    return out

# ================================================================================================================

# top 2 best performing models: ckpt-3500, ckpt-2000
chkpt_num = 3500
cwd = os.path.join(os.getcwd(),f'ft_mistral_7B_test/checkpoint-{chkpt_num}')

print('Loading model and tokenizer...')
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cwd, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(cwd, padding_side="left")
tokenizer.padding_side = 'left'

print('Creating eval data...')
# load eval paths
with open('eval_paths.txt','r') as f:
    data_paths = [i.strip() for i in f.readlines()]

print(data_paths)

data_paths = data_paths[:100] # 2000 tasks

prompts = []
answers = []
predicted = []
x = len(data_paths)
for d, i in enumerate(data_paths): # prepare eval data
    
    print(d, '/',x)
    data_packet = open_data_packet(i)
    task_to_do = create_training_entry_from_packet(data_packet, find_keyframes_indexes(data_packet))
    
    for j in task_to_do:

        prompt, answer = create_eval_pair(j)
        prompts.append(prompt)
        answer = answer.replace('</s>', '')
        answers.append(answer)
        predicted.append(-1)
    
    prompts.append('/')
    answers.append('/')
    predicted.append('/')

max_batch_size = 10
last_delimeter = 0
current_batch = []

extra_prompts_length = len(prompts) % max_batch_size
main_batch = prompts[:len(prompts) - extra_prompts_length]
print('Prompt creation complete.')
print(f'main batch size:', len(main_batch))
print(f'extra batch size:', extra_prompts_length)

number_of_slashes = 0
for i in range(0, len(main_batch), max_batch_size):

    print(f'Executing main batch {chkpt_num}. Progress:', i, round(i/len(main_batch),2))
    batch = []

    for j in range(max_batch_size):
        
        if main_batch[i + j] == '/':
            number_of_slashes += 1
            continue

        else:

            batch.append(main_batch[i + j])

    responses = inference(model=model, prompts=batch)

    for response in responses:
        
        predicted[predicted.index(-1)] = response

if extra_prompts_length:

    print('Executing extra batch.')
    current_batch = prompts[len(prompts) - extra_prompts_length:]
    while '/' in current_batch: current_batch.pop(current_batch.index('/'))

    print('validating extra prompts...')
    responses = inference(model=model, prompts=current_batch)
    for response in responses:

        predicted[predicted.index(-1)] = response

with open(f'task_evaluation_{chkpt_num}.json','w') as f:

    f.write(json.dumps([prompts, answers, predicted]))