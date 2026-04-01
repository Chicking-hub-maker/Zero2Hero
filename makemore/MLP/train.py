import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os

# ==========================================
# 1. 超参数配置 (Hyperparameters)
# ==========================================
BLOCK_SIZE = 3         # 滑动窗口上下文长度 (记忆几个字符)
HIDDEN_SIZE = 100      # 隐藏层神经元数量
EMB_DIM = 2            # 词嵌入向量维度
BATCH_SIZE = 32        # Minibatch 大小
EPOCHS = 30000         # 训练总步数
LEARNING_RATE_1 = 0.1  # 初始学习率
LEARNING_RATE_2 = 0.01 # 衰减后的学习率
DECAY_STEP = 20000     # 学习率衰减的步数节点

# ==========================================
# 2. 数据加载与基建
# ==========================================
print("正在加载数据与构建词表...")
if not os.path.exists('names.txt'):
    raise FileNotFoundError("未找到 names.txt，请确保它在当前目录下！")

words = open('names.txt', mode='r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
VOCAB_SIZE = len(stoi)

# 构建数据集的函数
def build_dataset(words_list):
    X, Y = [], []
    for w in words_list:
        context = [0] * BLOCK_SIZE
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # 窗口滑动
    return torch.tensor(X), torch.tensor(Y)

# 打乱数据并划分为: 80% 训练集, 10% 验证集, 10% 测试集
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"训练集 (Train) 大小: X={Xtr.shape}, Y={Ytr.shape}")
print(f"验证集 (Dev)   大小: X={Xdev.shape}, Y={Ydev.shape}")
print(f"测试集 (Test)  大小: X={Xte.shape}, Y={Yte.shape}")

# ==========================================
# 3. 初始化神经网络参数
# ==========================================
print("\n正在初始化神经网络大脑...")
g = torch.Generator().manual_seed(2147483647)

C  = torch.randn((VOCAB_SIZE, EMB_DIM), generator=g)
W1 = torch.randn((BLOCK_SIZE * EMB_DIM, HIDDEN_SIZE), generator=g)
b1 = torch.randn(HIDDEN_SIZE, generator=g)
W2 = torch.randn((HIDDEN_SIZE, VOCAB_SIZE), generator=g)
b2 = torch.randn(VOCAB_SIZE, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

print(f"模型总参数量: {sum(p.nelement() for p in parameters)}")

# ==========================================
# 4. 终极训练大循环
# ==========================================
print("\n🚀 开始 Minibatch 梯度下降训练...")
stepi = []
lossi = []

for i in range(EPOCHS):
    # 1. 构造 Minibatch (修复了之前的 Bug，现在只从训练集 Xtr 里抽)
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    
    # 2. 前向传播
    emb = C[Xtr[ix]] # [32, 3, 2]
    h = torch.tanh(emb.view(-1, BLOCK_SIZE * EMB_DIM) @ W1 + b1) # [32, 100]
    logits = h @ W2 + b2 # [32, 27]
    loss = F.cross_entropy(logits, Ytr[ix])
    
    # 3. 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 4. 参数更新与学习率衰减
    lr = LEARNING_RATE_1 if i < DECAY_STEP else LEARNING_RATE_2
    for p in parameters:
        p.data += -lr * p.grad
        
    # 5. 记录与打印日志
    stepi.append(i)
    lossi.append(loss.item())
    
    if i % 1000 == 0 or i == EPOCHS - 1:
        print(f"Step {i:5d}/{EPOCHS} | Minibatch Loss: {loss.item():.4f}")

# ==========================================
# 5. 绘制并保存 Loss 曲线 (服务器专用)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(stepi, lossi)
plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('loss_curve.png') # 保存为图片而不是 plt.show()
print("\n📈 训练损失曲线已保存为 'loss_curve.png'")

# ==========================================
# 6. 验证模型真实性能 (全局 Loss)
# ==========================================
print("\n👑 评估模型全局真实性能...")

# 评估函数 (使用 @torch.no_grad() 可以节省内存，因为评估不需要算梯度)
@torch.no_grad()
def split_loss(split_name, X, Y):
    emb = C[X]
    h = torch.tanh(emb.view(-1, BLOCK_SIZE * EMB_DIM) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(f"{split_name} 全局真实 Loss: {loss.item():.4f}")

split_loss('Train (训练集)', Xtr, Ytr)
split_loss('Dev   (验证集)', Xdev, Ydev)
split_loss('Test  (测试集)', Xte, Yte)

# ==========================================
# 7. 见证奇迹：让模型自己写名字
# ==========================================
print("\n✨ 神经网络生成的全新名字:")
g_sample = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(10): # 生成 10 个名字
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        # 推断模式的前向传播
        emb = C[torch.tensor([context])] # [1, 3, 2]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        # 抽卡
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
            
    print(''.join(itos[i] for i in out[:-1]).capitalize()) # 去掉末尾的 '.' 并首字母大写