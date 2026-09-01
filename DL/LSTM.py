import torch
import torch.nn as nn

# ---------------------- 超参数 ----------------------
input_size = 1     # 输入特征维度
hidden_size = 8    # 隐藏层维度 h_t,c_t的向量长度
num_layers = 1     # LSTM层数

# 定义LSTM层
lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True   # batch_first=True: shape [batch,序列长度,特征]
)

# 构造输入数据：batch=2，序列长度=5，每个时刻1维特征
x = torch.randn(2, 5, input_size)

# 前向传播
# output：保存每一步全部 h_t ，shape [batch, seq_len, hidden_size]
# (hn, cn): hn最后一步隐藏状态；cn最后一步细胞状态
output, (hn, cn) = lstm(x)

print("output shape(全部时刻h_t):", output.shape)
print("hn shape(最后一步h_t):", hn.shape)
print("cn shape(最后一步c_t):", cn.shape)
