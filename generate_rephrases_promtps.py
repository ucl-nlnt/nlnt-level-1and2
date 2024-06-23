from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, os
import uuid
from dataset_io import save_data_packet, open_data_packet
import progressbar

def generate_random_filename():

    random_filename = str(uuid.uuid4().hex)[:16]
    return random_filename

model_id = "microsoft/Phi-3-mini-128k-instruct"

bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    #load_in_4bit=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #quantization_config = bnb_config, # un-comment to quantize your model,
    trust_remote_code=True
)

subfolder = 'training_data_pre'

# load dataset
file_list = os.listdir(subfolder)

file_list = file_list

errors = 0
bar = progressbar.ProgressBar(max_value=len(file_list))
bar.start()

for curr,i in enumerate(file_list):

    packet = open_data_packet(os.path.join(subfolder,i))
    prompt = packet['natural_language_prompt']

    sys_prompt = "You are a young, creative roboticist named Coline that is eager to help create the next big thing."
    user_prompt_message = """Hey Coline, can you help me make a bunch of data points for our data set? The goal is to create paraphrasings for the tasks in our dataset. We're using Robotis' Turtlebot3. """
    user_prompt_message += "Right now, it can only move forward and NOT backwards and turn left and right. It's kind of like a floor-drone of sort. It can also take photos and videos, but is limited by a Raspberry Pi camera. "
    user_prompt_message += "I have some examples done already, and I was kind hoping that you help me add onto them."
    user_prompt_message += f"""
\nYou need to break-down the task first into easy-to-digest pieces, and then format it in such a way that it's easy to parse your answer with Python. Use the delimeters "<explanation_start>" and "<explanation_end>" for your explanation, and "<paraphrase_start>" and "<paraphrase_end>" for your answers.

<example>
Original prompt : "Show me how you can move three quarters of a foot diagonally to your right, then move yourself to your relative east about an eigth of a ruler, finally, get yourself one whole meter at 133 degrees rightward."

<explanation_start>

Prompt Analysis: The prompt is a multi-step instruction involving motion in different directions and units. It integrates both imperial and metric systems and uses angles for directional changes. Here's the breakdown:

    First Movement: "Move three quarters of a foot diagonally to your right"
        Distance: 0.75 feet
        Direction: Diagonally to the right implies movement along both x (east) and y (north) axes in the positive direction, usually at a 45-degree angle from the starting point.

    Second Movement: "Move yourself to your relative east about an eighth of a ruler"
        Distance: Assuming a standard ruler length of 12 inches (1 foot), an eighth of a ruler is 18 * 12 = 1.581 * 12 = 1.5 inches.
        Direction: East, implying movement along the x-axis in the positive direction.

    Third Movement: "Get yourself one whole meter at 133 degrees rightward"
        Distance: 1 meter
        Direction: 133 degrees from the north axis (east being 90 degrees and south being 180 degrees), so this points southeast but more towards the south.

<explanation_end>

Thus, an apt paraphrase is: <paraphrase_start> "Move a distance equal to 0.75 feet at a 45-degree angle to your initial position, then proceed 1.5 inches directly east. Following this, proceed southeast, setting your course to 133 degrees from north, covering a distance of 1 meter." <paraphrase_end>
</example>

Here's one of the prompts from our current dataset for you to process: {prompt}
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

    # specify terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|end|>')
    ]

    outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.01,
    top_p=1.0)

    # response is a list of token outputs that need to be decoded.
    response = outputs[0][input_ids.shape[-1]:]

    # final output just gives the response of the model
    final_output = tokenizer.decode(response, skip_special_tokens=False)
    print('===============================')
    print('ORIGINAL PROMPT:', prompt)
    print(final_output)
    bar.update(curr+1)
    
    try:

        paraphrase = final_output[final_output.index('<paraphrase_start>'):final_output.index('<paraphrase_end>')]
        explanation = final_output[final_output.index('<explanation_start>'):final_output.index('<explanation_end>')]

        packet['prompt_breakdown'] = explanation.replace('<explanation_start>','') # delete the headers
        packet['natural_language_paraphrase'] = paraphrase.replace('<paraphrase_start>','')
    
    except:
        print('#########################################')
        print('Error occured processing paraprhase data.')
        print('#########################################')
        errors += 1
        continue

    filename = os.path.join(os.getcwd(),'training_data_pre_rephrased',i.replace('.compressed','_rephrased.compressed'))

    save_data_packet(packet,filename)

print(f'Job complete. Total errors: {errors}/{len(file_list)}')
bar.finish()