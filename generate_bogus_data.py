from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, os
import uuid
from dataset_io import save_data_packet, open_data_packet
import progressbar
import copy

def generate_random_filename():

    random_filename = str(uuid.uuid4().hex)[:16]
    return random_filename

model_id = "microsoft/Phi-3-mini-128k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #load_in_4bit=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
    #quantization_config = bnb_config # un-comment to quantize your model
)

errors = 0
bar = progressbar.ProgressBar(max_value=2000)

generated_task_list = []

# specify terminators
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|end|>")
]

for i in range(2000):

    sys_prompt = "You are a young, creative roboticist named Coline that is eager to help create the next big thing."
    user_prompt_message = """Hey Coline, can you help me make a bunch of data points for our data set? The goal is to create reasonable but impossible or bogus tasks in our dataset so that the model knows it cannot do that. We're using Robotis' Turtlebot3. """
    user_prompt_message += "Right now, it can only move forwards (it can move backwards too but it has to pivot around first), and turn left and right. It's kind of like a floor-drone of sort. It can also take photos and videos, but is limited by a Raspberry Pi camera."
    user_prompt_message += """The Turtlebot3 has the following available sensors and/or devices onboard:
        1.) Lidar sensor
        2.) Odometer
        3.) IMU (Inertial Measurement Unit)
        4.) Rasberry Pi Camera v2.1 (Max resolution of 3280 x 2464 pixel static images, and video of up to 1920 x 1080p @ 30fps video.)
        5.) Speaker module.\n\n"""
    user_prompt_message += "I have some examples done already, and I was kind hoping that you help me add onto them."
    user_prompt_message += f"""
\nYou need to break-down the task first into easy-to-digest pieces, and then format it in such a way that it's easy to parse your answer with Python. Use the delimeters "<task_name_start>" and "<task_name_end>" for the task name, "<explanation_start>" and "<explanation_end>" for your explanation.

Focus on more mundane day-to-day tasks that should be possible to a typical robot as seen below, but is impossible for the Turtlebot3 because of hardware or software limitations.

Here are some examples to get you going:

<example1>
<task_name_start> Go up the stairs and head towards the first door you see. <task_name_end>

<explanation_start> Prompt Analysis: The prompt involves the robot or the agent moving up a flight of stairs. Given that the robot does not have any climbing capabilities (it only has wheels), it cannot accomplish this task. <explanation_end>
</example1>

<example2>
<task_name_start> Make me a pizza for me. <task_name_end>

<explanation_start> Prompt Analysis: This task requires the agent or the robot to have fine motor skills with some sort of appendage. In addition to that, access to a cooking facilities and methods are also needed. 
As things currently stand, the Turtlebot3 is neither equipped with a robot arm to manipulate nor does it have any access to any kitchen or cooking appliances. <explanation_end>
</example2>

From your previous iterations, you have the following: {generated_task_list}. Avoid creating samey or reiterations of your old prompts.

Create only one task for now, and I will give you feedback if that task is already accomplished or not.
"""
    
    messages = [
        {
            "role" : "system", "content" : sys_prompt
        },
        {
            "role" : "user", "content" : user_prompt_message
        }
    ]

    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.8,
    top_p=1.0)

    # response is a list of token outputs that need to be decoded.
    response = outputs[0][input_ids.shape[-1]:]

    # final output just gives the response of the model
    final_output = tokenizer.decode(response, skip_special_tokens=False)
    print('===============================')
    print(final_output)
    
    final_output_acceptable = False
    
    while not final_output_acceptable:

        try:

            task_name = final_output[final_output.index('<task_name_start>'):final_output.index('<task_name_end>')].replace('<task_name_start>','')
            explanation = final_output[final_output.index('<explanation_start>'):final_output.index('<explanation_end>')].replace('<explanation_start>','')
        
            if task_name in generated_task_list:

                print('---------------')
                print('Copy found in prompt list, regenerating.')
                print('---------------')

                messages_copy = copy.deepcopy(messages)
                
                messages_copy.append({
                    "role":"assistant","content":final_output.replace('<|end|>','')
                })

                messages_copy.append({
                    "role":"user", "content":"Can you please create a new prompt? That prompt / reason is already in the dataset. Use the same format please."
                })

                input_ids = tokenizer.apply_chat_template(
                messages_copy,
                add_generation_prompt=True,
                return_tensors="pt"
                ).to(model.device)

                outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=1.0)

                # response is a list of token outputs that need to be decoded.
                response = outputs[0][input_ids.shape[-1]:]

                # final output just gives the response of the model
                final_output = tokenizer.decode(response, skip_special_tokens=False)
                print('---------------')
                print('Regeneration:')
                print(final_output)
                print('---------------')

            else:

                generated_task_list.append(task_name)
                final_output_acceptable = True

                fname = generate_random_filename() + '_impossible.txt'
                data_path = os.path.join('impossible_and_unreasonable',fname)

                with open(data_path, 'w') as f:
                    f.write(task_name)
                    f.write('|||')
                    f.write(explanation)

                print(f'Wrote {data_path}.')
        except:

            print('#########################################')
            print('Error occured while generating data.')
            print('#########################################')
            errors += 1
            continue

    
    bar.update(i)



bar.finish()