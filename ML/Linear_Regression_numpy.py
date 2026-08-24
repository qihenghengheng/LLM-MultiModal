# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1.生成模拟数据集
np.random.seed(42)
m = 100  #样本数量
x = np.linspace(0, 10, m)
y = 2.5 * x + 4 + np.random.randn(m)*1.5  #真实w=2.5,b=4，加入噪声

# 初始化参数
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000

# 损失函数 MSE
def compute_loss(x, y, w, b):
    m = len(x)
    y_pred = w * x + b
    loss = np.sum((y_pred - y)**2) / (2*m)
    return loss

# 梯度下降训练
loss_history = []
for epoch in range(epochs):
    m = len(x)
    y_pred = w * x + b
    dw = np.sum((y_pred - y) * x) / m
    db = np.sum(y_pred - y) / m

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss = compute_loss(x,y,w,b)
    loss_history.append(loss)
    if (epoch+1) %100 ==0:
        print(f"epoch:{epoch+1:4d} | loss:{loss:.4f} | w={w:.3f}, b={b:.3f}")

# 绘图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(x,y,label="真实数据")
plt.plot(x, w*x+b, c="red", label=f"拟合 y={w:.2f}x+{b:.2f}")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss_history)
plt.title("loss下降曲线")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.show()