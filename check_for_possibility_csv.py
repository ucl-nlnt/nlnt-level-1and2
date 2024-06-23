from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, os
import uuid
from dataio import save_data_packet, open_data_packet
import progressbar
import time

def generate_random_filename():

    random_filename = str(uuid.uuid4().hex)[:16]
    return random_filename

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code = True,
    #quantization_config = bnb_config # un-comment to quantize your model; Only supports Nvidia GPUs
)

q = """
Vague:
small forward = 0.2
average forward = 0.8
big forward = 1.5
"""
prompt_list = []
for i in os.listdir("impossible_and_unreasonable"):

    path = os.path.join(os.getcwd(),"impossible_and_unreasonable",i)
    with open(path, 'r') as f:

        q = f.readlines()
        d = ""

        for strings in q:
            d += strings

        prompt_list.append(d.split('|||')[0])

import csv
with open("nlnt_impossible.csv",'r') as f:

    csv_reader = csv.reader(f)
    for i in csv_reader:
        prompt_list.append(i[0])

import random, json
# load dataset

random.shuffle(prompt_list)
errors = 0
bar = progressbar.ProgressBar(max_value=len(prompt_list))
bar.start()

iteration = 1
total_time_taken = 0.0

done_already = os.listdir('training_data_pre_rephrased')

for curr,i in enumerate(prompt_list):

    if i.replace('.compressed','_rephrased.compressed') in done_already:
        bar.update(curr+1)
        continue

    prompt = i
    packet = {'natural_language_prompt':i}

    sys_prompt = "You are a young roboticist that is currently helping in labeling a data set."
    user_prompt_message = """Can you help me make a bunch of data points for our data set? The goal is to determine if the tasks in our dataset are possible (or at least practical, safe, or ethical) by inducing chain-of-thought in the model that we're going to train. We're using Robotis' Turtlebot3. """
    user_prompt_message += "I have some examples done already, and I was kind hoping that you help me add onto them."
    user_prompt_message += f"""
\nYou need to break-down the task first into easy-to-digest statements and pieces of information. Use the delimeters "<explanation_start>" and "<explanation_end>" for your explanation and use "### Possibility: True" or "### Possibility: False" for your final answer.

If there are any vague units, for example "move a bit forwards", you can use the following shortlist as a substitute:

small vague movement forwards = 0.2 meters ("a small amount", "inch a petite distance")
average or medium-length movement forwards = 0.8 meters ("a normal amount", "an average amount")
long distance vague forward movement instruction = 1.5 meters ("a large distance", "a significant distance", "a vast distance")

The following are the current capabilities of the Turtlebot:

1.) Moving around on the floor.
    - It cannot move even over small obstacles
    - It can move forwards and turn left or right, but it cannot be made to move backwards by its own. It needs to turn around first and then move backwards.

2.) Scan the immediate area around it with a planar lidar sensor, up to a maximum of 3.5 meters away from the sensor.

3.) Take a photo (maximum resolution of 1080 x 1920) or a video (720 x 1280 @ 30fps).
    - The Turtlebot3 can take a single photo or start a video stream.
    - Cannot take video in the dark.
    - Cannot be swiveled up and down, and is dependent on the robot to be rotated.

4.) Record audio.
    - Also supported while taking a video stream.

5.) Odometer

6.) IMU

Include a list of sensors and/or features that you think will be useful in a JSON. Indicate it with "<device_flags_start>" and "<device_flags_end>".

The following device flags are available:

1.) take_photo : takes a photo
2.) livestream : sends a livestream of data frames to the server that the robot is connected to
3.) take_video : records a video of the activity
4.) audio_record : records audio
5.) audio : records audio data with either a video or a livestream; defaults to false so you need to indicate it if it needs to be used
6.) no_special_features : used to signal that the command doesn't need the robot to use any of its special sensors or features

The following actions are not supported:

1.) Object manipulation.
    - The Turtlebot3 does not have an arm or any of the sort.

2.) Streaming media to and from the internet.
    - The Turtlebot3 does not have access to the internet

NOTES:
    - Assume that pattern-creation tasks is to be done on the surface that the robot is on. For example, making a square or star pattern is done by moving on top of the floor and not by actually using and art supplies.
    - Do not record photos, videos, or audio unless otherwise stated by the prompt.
    - the Turtlebot3 cannot draw. It cannot manipulate stationery or pens or any objects that can be used to mark.
    - assume that the Turtlebot3 is fragile and does not have any water nor shock resistance.
    - the Turtlebot3 is a small robot, about 150mm by 150mm in length and width, and about 200mm in height.
    - the current system needs an active internet connection to function; the Turtlebot3 functions of 2.4GHz band WiFi.
    - the TurtleBot3 cannot read text on its own, nor process vision data without the help of an AI model. Assume that the robot does not have an AI model to help it do these tasks.

Here are some examples for your reference:

<example1> (All-Possible Example)
Original prompt : "Show me how you can move three quarters of a foot diagonally to your right, then move yourself to your relative east about an eigth of a ruler, finally, get yourself a reasonable distance away at 133 degrees rightward."

<explanation_start>

Prompt Analysis: The prompt is a multi-step instruction involving motion in different directions and units. It integrates both imperial and metric systems and uses angles for directional changes. Here's the breakdown:

    Statment 1: "Move three quarters of a foot diagonally to your right"
        Vague: False
        Angular Displacement: approx. 45 degrees clockwise/right.
        Distance: 0.75 feet
        Direction: Diagonally to the right implies movement along both x (east) and y (north) axes in the positive direction, usually at a 45-degree angle from the starting point.
        Possibility: True

    Statement 2: "Move yourself to your relative east about an eighth of a ruler"
        Vague: False
        Angular Displacement: "relative east" means to the right of the Turtlebot3 at that moment.
        Distance: Assuming a standard ruler length of 12 inches (1 foot), an eighth of a ruler is 18 * 12 = 1.581 * 12 = 1.5 inches.
        Direction: East, implying movement along the x-axis in the positive direction.
        Possibility: True

    Statement 3: "Get yourself one whole meter at 133 degrees rightward"
        Vague: True
        Possibility: True
        Angular Displacement: 133 degrees clockwise
        Distance: possibly 0.8 meters
        Direction: 133 degrees from the north axis (east being 90 degrees and south being 180 degrees), so this points southeast but more towards the south.
        Possibility: True

    Given that it's possible to perform these actions, the task at hand is therefore:
    ### Possibility: True

    <device_flags_start> {{"no_special_features" : True}} <device_flags_end>

<explanation_end>

</example1>

<example2> (Partially vague and possible but nevertheless incompleteable)
<explanation_start>
Original prompt : "Please make your way up the stairs and find the kitchen. Once there, create a pepperoni pizza."

<explanation_start>

Prompt Analysis : The Turtlebot3 cannot follow this prompt. It does not have the capabilties to mobilize itself up a flight of stairs.

    Statement 1: "Make your way up the stairs."

        Vague: False
        Angular Displacement: None.
        Distance: Unspecified, but it is assumed that the stairs are of a reasonable length.
        Direction: Upwards.
        Possibility: False

    Statement 2: "Find the kitchen."

        Notes: This instruction may be possibile assuming that there is a kitchen in the upper floor.
        Vague: True
        Angular Displacement: Unknown
        Distance: Unknown
        Possibility: Unknown
        
    Statement 3: "Create a pepperoni pizza."

        Notes: The Turtlebot3 cannot manipulate objects as it does not have an arm.
        Vague: True (Instructions are clear, but details of the pizza such as size is unknown)
        Possibility: False

    The following instruction is at least partially impossible to accomplish given the capabilities of the Turtlebot3 robot, and thus:
    ### Possibility: False
    <device_flags_start> {{"no_special_features" : True}} <device_flags_end>

<explanation_end>
</example2>

<example3> (Using special features)
<explanation_start>
Original prompt : "Please look for my cat, she's a cute and fat orange tabby named Marshmallow. She should be by the back door in the kitchen. Take a short video of her for me just so I know she's safe."

<explanation_start>

Prompt Analysis : The Turtlebot3 cannot follow this prompt. It does not have the capabilties to mobilize itself up a flight of stairs.

    Statement 1: "Please look for my fat oranged tabby cat named Marshmallow."

        Vague: False
        Distance: Unknown
        Direction: Unknown
        Possibility: Unknown (True, assuming that such a cat exists within the close vicinity.)

    Statement 2: "She should be by the back door in the kitchen."

        Notes: This indicates where the cat may possibly be.
        Vague: True
        Angular Displacement: Unknown
        Distance: Unknown
        Possibility: Unknown (True, assuming that the cat is actually by the back door in the kitchen.)
        
    Statement 3: "Take a short video of her for me."

        Notes: The length of the video is unspecified, so it can be assumed that the length is reasonably somewhere between 10 and 20 seconds.
        Vague: True (Instructions are clear, but the length is unknown)
        Possibility: Unknown (It's not known if Marshmallow is actually at that spot in the kitchen.)

    The following instruction is at least partially impossible to accomplish given the capabilities of the Turtlebot3 robot, and thus:
    ### Possibility: True
    <device_flags_start> {{"take_video" : True, "audio" : True}} <device_flags_end>

<explanation_end>
</example3>

Once again, use the delimeters "<explanation_start>" and "<explanation_end>" for your explanation and use "### Possibility: True" or "### Possibility: False" for your final answer. Be sure to add all your answers before the <explanation_end>, especially the Possibility.
Be binary with your Possibility indication. Answer it as a whole, i.e., can the entire prompt be completed.
Now that that's done, it's your turn.

Here's one of the prompts from our current dataset for you to process: <prompt> {prompt} </prompt>. If the prompt is excessively unintelligible or just immoral or toxic, indicate it as ### Possibility: False.
"""
    messages = [
        {
            "role" : "system", "content" : sys_prompt
        },
        {
            "role" : "user", "content" : user_prompt_message
        }
    ]

    t = time.time()
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)

    # specify terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|end|>") # phi3
        #tokenizer.convert_tokens_to_ids("<|eot_id|>"), # llama3
    ]

    outputs = model.generate(
    input_ids,
    max_new_tokens=1024,
    eos_token_id=terminators)

    # response is a list of token outputs that need to be decoded.
    response = outputs[0][input_ids.shape[-1]:]

    # final output just gives the response of the model
    final_output = tokenizer.decode(response, skip_special_tokens=False)
    print('===============================')
    print("PROMPT:", prompt)
    print(final_output)
    bar.update(curr+1)
    
    try:

        explanation = final_output[final_output.index('<explanation_start>'):final_output.index('<explanation_end>')]
        packet['explanation'] = explanation

    except:
        print('#########################################')
        print('Error occured processing paraprhase data.')
        print('#########################################')
        errors += 1
        continue

    filename = os.path.join(os.getcwd(),'impossible_rephrased',generate_random_filename() + '.txt')

    with open(filename,'w') as f:
        f.write(json.dumps(packet))

    total_time_taken += time.time() - t
    print('\n')
    print(f"### Average time per prompt: {round(total_time_taken / iteration, 3)}")
    print('\n')
    iteration += 1

print(f'Job complete. Total errors: {errors}/{len(prompt_list)}')
bar.finish()