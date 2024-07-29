#from pygpt4all.models.gpt4all import GPT4All
from gpt4all import GPT4All
#from langchain.llms import GPT4All
def new_text_callback(text):
    print(text, end="")

model = GPT4All("d:\\data\\Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model.generate("Once upon a time, ", n_predict=55, callback=new_text_callback)
