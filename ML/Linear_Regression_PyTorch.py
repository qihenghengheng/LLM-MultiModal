import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子
torch.manual_seed(42)

# 1.构造数据集
m = 100
x = torch.linspace(0,10,m).unsqueeze(1)  # shape [100,1]
y = 2.5 * x + 4 + torch.randn_like(x)*1.5

# 2.定义模型：线性层 y = w*x + b
model = nn.Linear(in_features=1, out_features=1)

# 3.损失函数、优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
loss_list = []

# 4.训练循环
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    # 反向传播+更新参数
    optimizer.zero_grad()  #梯度清零，非常关键！
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if (epoch+1) %100 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"epoch:{epoch+1:4d} loss:{loss:.4f}  w={w:.3f}, b={b:.3f}")

# 获取训练完的参数
w_trained = model.weight.item()
b_trained = model.bias.item()

#绘图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(x.numpy(), y.numpy(), label="原始数据")
plt.plot(x.numpy(), model(x).detach().numpy(), c="red", label=f"拟合 y={w_trained:.2f}x+{b_trained:.2f}")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss_list)
plt.title("Loss曲线")
plt.show()