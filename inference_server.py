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
from KNetworking import DataBridgeServer_TCP
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gc

is_generating = False

class inferenceServer:

    def __init__(self, model_name):

        self.model = None
        self.tokenizer = None
        self.loaded_model_name = model_name

        self.message_buffer = []
        self.load_model()
        self.load_tokenizer()
        
    def inference(self, message_batch:list = [-1]):

        try:
        
            t_start = time.time()
            inputs = self.tokenizer(message_batch, padding=True, add_special_tokens=True, return_tensors='pt', max_length=700).to('cuda')
            generate_ids = self.model.generate(**inputs, max_length=700)
            out = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
            delta_t = time.time() - t_start
        
        except Exception as e:
            return (0, e, -1)
        
        return (1, out, delta_t)
    
    def load_model(self):

        print('Loading model...')
        try:

            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            self.model = AutoModelForCausalLM.from_pretrained(self.loaded_model_namen, device_map = "auto")
            self.load_tokenizer(self.loaded_model_name)

        except Exception as e:
            print(e)
            return (0, e)

        print(f'Model loaded. {self.loaded_model_name}')
        return (1, f"Loaded model {self.loaded_model_name}")

    def load_tokenizer(self, tokenizer_version = 'v1/full'):

        print('setting tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
        self.tokenizer.padding_side = 'left'
        print('tokenizer set.')

        return (1, tokenizer_version)
    
server = inferenceServer('v1/full')
app = FastAPI()

class Prompt(BaseModel):

    content: str

class Generated(BaseModel):

    generated: str

@app.get("/")
async def hello():
    return {"message": "hello world!"}

@app.post("/send-prompt/")
async def receive_prompt(prompt: Prompt):

    global is_generating, server

    if is_generating: return {"generated":"server is busy"}

    data = prompt.content

    # if 'setModel' in data:
        
    #     try:

    #         data = data.split('=')
    #         model = data[1]

    #     except:
    #         return {"generated":"invalid model"}
        
    #     code, text = server.load_model(model)
    #     server.load_tokenizer(model)

    #     return {"generated":str(text)}

    is_generating = True
    response = server.inference(message_batch=[prompt.content])[1][0]
    is_generating = False

    return {"generated": response}

# while True:
#     time.sleep(1)