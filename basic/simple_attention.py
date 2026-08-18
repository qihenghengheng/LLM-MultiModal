import torch
# 模拟输入文本向量
batch, seq_len, dim = 1, 3, 4
x = torch.randn(batch, seq_len, dim)

# 初始化Q/K/V权重
Wq = torch.nn.Linear(dim, dim)
Wk = torch.nn.Linear(dim, dim)
Wv = torch.nn.Linear(dim, dim)

Q = Wq(x)
K = Wk(x)
V = Wv(x)

# 计算注意力分数
attn_score = torch.matmul(Q, K.transpose(-1, -2))
attn_weight = torch.softmax(attn_score, dim=-1)
output = torch.matmul(attn_weight, V)

print("注意力权重矩阵：\n", attn_weight)
print("注意力输出：\n", output)