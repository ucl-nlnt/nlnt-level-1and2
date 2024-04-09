import os
import zlib, json
import quart_funcs
import random

""" Data packet structure
    {
    username : ...,
    natural_language_prompt : ...,
    timestamp_s : ...,
    timestamp_float : ...,
    ground_truth : ...,
    simulation : int [optional],
    states : [
        {
            laser_scan : {
                None
                NOTE: currently unsupported in level 1 and 2 simulated data 
            },

            twist : {
                linear : [x, y, z] # usually x is the only non-zero
                angular : [x, y, z] # usually z is the only non-zero
            },

            imu : {
                quarternion_orientation : [...],
                orientation_covariance : [...],
                angular_velocity : [x, y ,z],
                angular_velocity_covariance : [...],
                linear_acceleration : [...],
                linear_acceleration_covariance : [...]
            },

            odometry : {
                time_sec : float,
                time_nano : float,
                pose_position : [x, y, z],
                pose_orientation_quarternion : [x, y, z, w],
                object_covariance : float array [...] # usually not useful
            },

            battery : { # not usually useful
                ...
            },

            frame_data : np.ndarray # turtlebot 3 camera image; NOTE: None for simulated as of april 8, 2024
            distance_traveled : float,
            radians_rotated : float,

        },
        {
        ...
        },
        ...
    ]


    }
"""

def get_list_of_files(folder):

    p = os.getcwd()
    folder_path = os.path.join(p, folder)
    return [os.path.join(folder_path,i) for i in os.listdir(folder_path)]

def open_data_packet(path):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

def find_keyframes_indexes(state_packet):

    frames = state_packet['states']

    twist_last = frames[0]['twist']
    keyframes = [0]
    for i,frame in enumerate(frames):
        if frame['twist'] != twist_last:
            twist_last = frame['twist']
            keyframes.append(i)
        
    return keyframes

def prep_frame_data(state_list, keyframe_indexes):

    data = [state_list[i] for i in keyframe_indexes]

    # shift and rotate odometry data
    
    reference_position = data[0]['odometry']['pose_position']
    reference_rotation = data[0]['odometry']['pose_orientation_quarternion']


    inverse_reference_rotation = quart_funcs.inverse_quarternion(reference_rotation)

    for i, frame in enumerate(data):
        
        odom = frame['odometry']
        old_position, old_rotation = odom['pose_position'], odom['pose_orientation_quarternion']

        new_position = quart_funcs.adjust_position_origin(reference_position, old_position)
        new_rotation = quart_funcs.adjust_orientation_origin(inverse_reference_rotation, old_rotation)

        odom['pose_position'] = new_position
        odom['pose_orientation_quarternion'] = new_rotation

        data[i]['odometry'] = odom
    
    return data

def create_training_entry_from_packet(data_packet, keyframe_indexes, bos = '<bos>', eos = '<eos>'):

    # Gemma: {'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
    # Mistral: // to be filled out later
    # Llama: // to be filled out later

    prompt = data_packet['natural_language_prompt']
    kstates = prep_frame_data(data_packet['states'], keyframe_indexes)

    training_examples = []

    entry = f"{bos} [cmd] {prompt} [cmd_end]"
    state_number = 0
    for i in range(1,len(kstates)):
        
        entry += ' [OBS] '

        current_kframe = kstates[i -1]
        next_kframe = kstates[i]

        # limiting the numerical precision to two decimal places to avoid wasting context
        # NOTE: this algorithm assumes that there are at least two states in the data packet

        # I try to not use words that are present in the natural language dataset for the state keys
        t = float(str(current_kframe['odometry']['time_sec']) + '.' + str(current_kframe['odometry']['time_nano']))
        t_next = float(str(next_kframe['odometry']['time_sec']) + '.' + str(next_kframe['odometry']['time_nano']))
        
        dt = round(t_next - t, 3)

        entry += str(
            
            {
                'state number' : state_number,
                'message' : current_kframe['twist'], 
                'dt' : dt
            }

        )

        state_number += 1
    
    training_examples.append(entry + ' ' + eos) # add to training examples

    return training_examples

def read_and_process_all(data_path, bos='<|endoftext|>', eos='<|endoftext|>'):

    entries = []

    paths = get_list_of_files(data_path)

    for path in paths:
        packet = open_data_packet(path)
        training_entries = create_training_entry_from_packet(packet, find_keyframes_indexes(packet), bos, eos)
        entries += training_entries

    print("number of training data entries:", len(entries))
    return entries

entries = read_and_process_all('training_data_pre', bos="<s>", eos="</s>")
random.shuffle(entries)
entries = entries[:5000] # limit data set count
# print(entries[0])

from transformers import AutoTokenizer

# Replace 'your_dataset_name' with the actual name of your dataset
# and 'your_dataset_field' with the field name containing the text you want to tokenize

# Initialize the tokenizer
# Replace 'your_model_name' with the actual model name you're using, e.g., "bert-base-uncased"
tokenizer_name = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Function to calculate the number of tokens in each entry
def count_tokens(example):
    return len(tokenizer.tokenize(example))

tokens = [count_tokens(i) for i in entries]

# Apply the function across the dataset to find the max number of tokens
# Note: Depending on the size of your dataset, this operation might take some time.
max_tokens = max(tokens)
print("Maximum tokens in a dataset entry:", max_tokens)

with open(os.path.join('training_data_post','train','data.json'),'w') as f:
    f.write(json.dumps(entries))