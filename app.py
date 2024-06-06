import os
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig

base_path = './internlm2-chat-1_8b' # 已W4A16量化
os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(session_len=8192) 
pipe = pipeline(base_path, backend_config=backend_config)

def model(text,*args):
    response = pipe(text).text
    return response

demo = gr.ChatInterface(fn=model, textbox=gr.Textbox(), chatbot=gr.Chatbot())
demo.launch()
