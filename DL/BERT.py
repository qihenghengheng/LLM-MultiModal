import os
# hf国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertTextClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = outputs.pooler_output
        logits = self.classifier(cls_out)
        return logits


model = BertTextClassifier(num_classes=2)

text = "I love this book!"
enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
print("分类logits输出：", logits)
pred = torch.argmax(logits, dim=1)
print("预测类别：", pred.item())
