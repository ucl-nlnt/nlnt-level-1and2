from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import time
import sys
import math
import ast
import random
import zlib
import torch
import threading

bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# cwd = 'meta-llama/Meta-Llama-3-8B'
cwd = 'meta-llama/Meta-Llama-3-8B-Instruct'
# cwd = 'mistralai/Mistral-7B-Instruct-v0.2'
def open_data_packet(path):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

def save_data_packet(data:dict, path):

    with open(path, 'wb') as f:

        f.write(zlib.compress(json.dumps(data).encode('utf-8')))

packet_paths = os.listdir("training_data_pre")

def create_rephrase_instance(cwd: str, packet_paths, thread_num: int = 0):

    tokenizer = AutoTokenizer.from_pretrained(cwd, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cwd, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    print(tokenizer.special_tokens_map)

    def inference(prompt: str):

        t_start = time.time()
        inputs = tokenizer([prompt], return_tensors='pt', max_length=2000).to('cuda')

        generate_ids = model.generate(**inputs, max_length=2500)
        out = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
        delta_t = time.time() - t_start
        print('Batch inference took', round(delta_t,2), 'seconds')
        return out

    for j, i in enumerate(packet_paths):

        packet = open_data_packet(os.path.join('training_data_pre', i))

        prompt = '<|begin_of_text|> ' if cwd == 'meta-llama/Meta-Llama-3-8B-Instruct' else '<s> '
        prompt += f"""Task: Rephrase the following prompt for inclusion in a dataset. Decompose the prompt into its essential elements, then use these elements to construct a new prompt. The original prompt is: "{packet['natural_language_prompt']}". Please provide only one, and repeat only ONE, alternative version of this prompt and separate it with delimeters. Make the response string easy to parse and separate with Python."""
        prompt += '\n'
        prompt += f"""
# Here are some examples for you:
        
<user_example1>
    original prompt: "Turn 90 degrees CW and then move by 3/4ths of a meter."

    step-by-step explanation:

    1.) The actor rotates right by 90 degrees clockwise.
    2.) Pause to verify the orientation.
    3.) Move forward a distance of 0.75 meters.
    4.) Stop at the goal and wait for further instructions.

    answer:
    <start_answer> "Please turn to your right by 90 degrees and then proceed to the point approximately 0.75 meters away from your current position." <end_answer>
</user_example1>

<user_example2>
    original prompt: "Advance for a total distance of 4.56 feet to your relative west. Also, move three eighths of a ruler diagonally to your right."

    step-by-step explanation:

    1.) Move west for a distance of about 4.56 feet (1.39 meters).
    2.) Additionally, move diagonally to your right by 4.5 inches (approximately 114 millimeters).

    answer:
    <start_answer> "Proceed about 4.56 feet westward and then shift diagonally to your right by approximately 4.5 inches." <end_answer>
</user_example2>

<user_example3>
    original prompt: "Turn 68% CCW 2x, then get yourself around 1 ruler forward."

    step-by-step explanation:

    1.) The agent is instructed to turn counterclockwise by 136% of a full turn, approximately 244.8 degrees.
    2.) After turning, the actor is instructed to move forward roughly 12 inches.

    answer:
    <start_answer> "Turn counterclockwise by about 245 degrees and then move forward about the length of a ruler, which is approximately one foot." <end_answer>
</user_example3>

<user_example4>
    original prompt: "Make a quarter turn to the left, then step forward two paces."

    step-by-step explanation:

    1.) The actor is instructed to make a 90-degree turn to the left.
    2.) Take two steps forward, each step traditionally being about 30 inches based on an average stride.

    answer:
    <start_answer> "First, make a quarter turn to your left and then take two steps forward, with each step measuring roughly two and a half feet." <end_answer>
</user_example4>

<user_example5>
    original prompt: "Hop twice, then spin around and sit down."

    step-by-step explanation:

    1.) The actor is instructed to perform two quick hops.
    2.) Immediately after hopping, spin around 360 degrees.
    3.) Conclude the sequence by sitting down on the ground.

    answer:
    <start_answer> "Please hop twice, perform a full spin, and then sit down on the floor." <end_answer>
</user_example5>


# Now, it's your turn. Write your answer below.

<assistant>
    original prompt: "{packet['natural_language_prompt']}"

    step-by-step explanation:

    1.)"""

        print('===================================================')
        print(f'Iteration (thread {thread_num}):', j,'/',len(packet_paths))

        response = inference(prompt)[0]
        print(response)
        response = response.replace('</s>', '')

        packet['generated_rephrasal'] = response

        save_data_packet(packet, os.path.join('training_data_pre_processed',i))

#create_rephrase_instance(cwd, packet_paths[:len(packet_paths) // 2], 1) # thread 1
create_rephrase_instance(cwd, packet_paths[len(packet_paths) // 2:], 2) # thread 2