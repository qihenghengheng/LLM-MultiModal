# Day5 LoRA轻量化微调项目
## 项目简介
使用Qwen大模型，采用LoRA低秩适配微调，在自制AI知识对话数据集做微调。
冻结预训练大模型主干，仅训练少量低秩矩阵，降低显存开销。

## 文件说明
- dialogue_data.json：自制对话数据集
- lora_train.py：LoRA训练脚本，4bit量化加载模型
- lora_infer.py：加载LoRA适配器推理对话

## 运行命令
```bash
#训练
python lora_train.py
#推理对话测试
python lora_infer.py
