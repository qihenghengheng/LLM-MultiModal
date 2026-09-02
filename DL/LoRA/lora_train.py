import os
# hf国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ==========配置区，显存小就把模型改成 Qwen‑1.8B‑Chat ==========
model_name = "Qwen/Qwen-7B-Chat"
# lora超参
lora_rank = 8
lora_alpha = 8

# 1.加载数据集
dataset = load_dataset("json", data_files="./LoRA/dialogue_data.json", split="train")


# 2.加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 构造Qwen对话prompt模板
def format_prompt(sample):
    prompt = f"""<|im_start|>user
{sample['instruction']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""
    return {"text": prompt}

dataset = dataset.map(format_prompt)

def tokenize_fn(sample):
    return tokenizer(
        sample["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# 3.加载模型，load_in_4bit开启4bit量化，大幅降低显存占用
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True,
    device_map="auto"
)

# 4.配置LoRA
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() #打印可训练参数占比，应该只有百分之零点几

# 5.训练参数
training_args = TrainingArguments(
    output_dir="./lora_adapter",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_dir="./train_logs",
    logging_steps=2,
    save_strategy="epoch",
    report_to="none"
)

# 6.Trainer开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
print("LoRA适配器训练完成，保存在 ./lora_adapter")
