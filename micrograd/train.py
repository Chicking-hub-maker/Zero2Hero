import matplotlib.pyplot as plt
from micrograd import MLP

# 1. 准备微型数据集
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -2.0, 1.0]

# 2. 实例化神经网络 (3个输入，两层4神经元的隐藏层，1个输出)
n = MLP(3, [4, 4, 1])

# 3. 超参数设置
epochs = 100           
learning_rate = 0.05  

# >>>  准备一个空列表，用来收集每一步的 loss <<<
losses = []

print("--- 开始训练 ---")

for k in range(epochs):
    # 前向传播
    ypred = [n(x) for x in xs]
    
    # 计算损失 (MSE)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    losses.append(loss.data)
    
    # 梯度清零
    for p in n.parameters():
        p.grad = 0.0
        
    # 反向传播 
    loss.backward()
    
    # 参数更新 (梯度下降)
    for p in n.parameters():
        p.data += -learning_rate * p.grad
        
    print(f"Epoch {k+1:2d} | Loss: {loss.data:.4f}")

print("\n--- 训练结束，验证最终预测结果 ---")
for i, (ygt, yout) in enumerate(zip(ys, ypred)):
    print(f"样本 {i+1} -> 目标值: {ygt:4.1f} | 模型预测: {yout.data:.4f}")

# 绘制学习曲线 <<<
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), losses, marker='o', linestyle='-', color='b', markersize=4)
plt.title("Neural Network Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Loss)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()