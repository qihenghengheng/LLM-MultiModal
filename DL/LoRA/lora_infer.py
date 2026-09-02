import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-7B-Chat"
lora_path = "./lora_adapter/checkpoint-5"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_path)

def chat(query):
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res

if __name__ == "__main__":
    print(chat("请解释什么是LoRA微调"))
    print("-"*30)
    print(chat("BERT和GPT有什么区别"))
