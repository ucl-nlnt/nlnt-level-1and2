## with CSS & livestreaming (WORKS !)
# documentation for livestreaming: https://www.gradio.app/guides/reactive-interfaces

import gradio as gr
from transformers import pipeline
import numpy as np
from inference_gradio import main
import json

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

theme = gr.themes.Default(primary_hue= gr.themes.colors.emerald, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate).set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_200",
)

css = """
.color_btn textarea  {background-color: #228B22; !important}
"""

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def nlnt (vid_check, prompt, video, history="None"):
  if vid_check == True:
        return level3_model(prompt, video)
  else:
        #You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: Please get yourself rightwards by about six full centimeters, then coast to your right  a whole meter, then advance nine full centimeters backwards.. | History: [ None ] ### Answer:
        #add_prompt = "You are given the task to act as a helpful agent that pilots a robot. Given the the frame history, determine the next frame in the series given the prompt and the previous state. Expect that any given data will be in the form of a JSON, and it is also expected that your reply will be also in JSON format. Set the 'completed' flag to '#complete' when you are done, otherwise leave it as '#ongoing'. Here is your task: " + prompt + " | History: [ " + history + " ] ### Answer:"
        x = json.dumps(main(prompt))
        x_dict = json.loads(x)
        
        #print(x)

        returned = json.loads(x)
        history = []

        while returned["instruction complete"] == "#ongoing":       # JSON bug here where it doesn't recognize dictionary
            # send x to Turtlebot using TCP => Turtlebot churva
            # receive y from turtlebot using TCP

            history.append(y[:])
            x = main(prompt, history)

            #print(x)
        
        return x_dict
    #main(prompt)
      #return level2_model(prompt)

def level2_model (prompt):
    return "level 2: " + prompt

def level3_model (prompt, video):
    return "level 3: " + prompt

def show_vid (vid_check):
    if vid_check:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)

with gr.Blocks(theme=theme, css=css, title = "NLNT Demo") as demo:
    gr.Markdown(
    """
    # Natural Language Ninja Turtle
    A Natural Language to ROS2 Translator for the Turtlebot V3 Burger
    """)
    with gr.Row():
        vid_check = gr.Checkbox(label = "connect live video")
    with gr.Row(equal_height=True):
        audio = gr.Audio(sources=["microphone"])
        prompt = gr.Textbox(label = "Instruction", placeholder = "move 1.5 meters forward", interactive = True)
    with gr.Row():
        clr_audio = gr.ClearButton(value = "clear audio", components = [audio])
        transcribe_btn = gr.Button(value = "Transcribe", elem_classes = "color_btn")
        clr_text = gr.ClearButton(value = "clear text", components = [audio, prompt])
        transcription = transcribe_btn.click(fn=transcribe, inputs=audio, outputs=prompt)
        #transcription = gr.Interface(transcribe, audio, prompt)
    with gr.Row():
      ttbt_btn = gr.Button(value = "Run Instruction", elem_classes = "color_btn")
    with gr.Row():
        video = gr.Image(sources=["webcam"], streaming=True, visible=False)
        ckbx = vid_check.select(fn = show_vid, inputs = vid_check, outputs = video)
        #vid_check.change(show_vid, vid_check, video)
    with gr.Row():
      with gr.Column():
        status = gr.Textbox(label = "Status", placeholder = "Please enter your prompt.")
        #history = None
        run_nlnt = ttbt_btn.click(fn=nlnt, inputs=[vid_check, prompt, video], outputs=status)


demo.launch()
