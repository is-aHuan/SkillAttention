import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = 'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'
model = AutoModelForCausalLM.from_pretrained(hf_path, 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True)
model.cuda("cuda:1")

config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length, padding_side = config.tokenizer_padding_side)
# prompt="What are these?"
# image_url="http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
task = "Put the apple in the basket"
image = "/root/codespace/SkillDiffuserTune/assets/planning.png"
prompt = (
            f"Given the objects in the image, if you are required to complete the task \"{task}\", "
            "what is your detailed plan? Write your plan and explain it in detail, using the following format: "
            "Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n"
        ) 
output_text, genertaion_time = model.chat(prompt=prompt, image=image, tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
