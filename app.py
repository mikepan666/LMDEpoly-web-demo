import os
import gradio as gr
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig
from typing import Generator, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


print("lmdeploy version: ", lmdeploy.__version__)
print("gradio version: ", gr.__version__)


# clone 模型
MODEL_PATH = './models/internlm2-chat-1_8b'
os.system(f'git clone -b master https://code.openxlab.org.cn/mikepan666/lmdepoly.git {MODEL_PATH}')
os.system(f'cd {MODEL_PATH} && git lfs pull')


tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8b",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
