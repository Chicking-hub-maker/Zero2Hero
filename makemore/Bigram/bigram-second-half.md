# Bigram 神经网络建模

## 统计建模的一些问题
没有经过训练的模型，输出纯粹是垃圾，故而现在需要转向神经网络的方法进行设计

## 准备数据集
- 输入特征+目标标签
- 对应到Bigram名字生成的任务中，逻辑就是**看见前一个字母（输入），预测下一个字母（目标）**
```python
import torch

xs,ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2] #字符转为数字索引

        xs.append(ix1)  #分别放进输入集和目标集
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


print(f"数据总量：{xs.nelement()}个字符")
print("xs(输入):",xs[5:])
print("ys(目标):",ys[5:])
```
执行完这段代码，将会得到两个包含了几十万个数字的巨大一维张量 xs 和 ys。这就是我们接下来要喂给神经网络的全部“教材”

### 问题：索引编码作为字符的标签是否合理
如果我们直接把 `13`（代表 `'m'`）和 `5`（代表 `'e'`）喂给神经网络，模型在做乘法时会天然地认为 $m = 2.6 \times e$。这在数学上是极其荒谬的，因为字母之间根本没有大小和倍数关系。  
**我们应该构建完整的平等的类别变量**

### One-Hot编码
这是处理分类数据的标准操作
```python
import torch.nn.functional as F

#将输入序列处理为One Hot编码
xenc = F.one_hot(xs,27).float()

print("One-Hot 编码后的张量形状:", xenc.shape)
print("原本的数字 xs[0]:", xs[0].item())
print("One-Hot 编码后的 xs[0]:", xenc[0])
```
```
One-Hot 编码后的张量形状: torch.Size([228146, 27])
原本的数字 xs[0]: 0
One-Hot 编码后的 xs[0]: tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

## 前向传播
```python
#1、初始化神经网络权重矩阵
g = torch.Generator().manual_seed(2147483647)#随机种子，确保生成的内容一样
W = torch.randn((27,27),generator=g,requires_grad=True)

#2、矩阵乘法
logits = xenc @ W

#3、得分转概率
counts = logits.exp()
probs = counts / counts.sum(1,keepdims = True)

print("预测概率 probs 的形状:", probs.shape)
print("第一行的概率和:", probs[0].sum().item()) # 应该完美等于 1.0
```

## 反向传播与NLL
```python 
import torch.nn.functional as F

# 1. 初始化网络大脑
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 统计一下我们到底有多少个字符对样本
num_examples = xs.nelement()
print(f"总样本数: {num_examples}")

epochs = 200
learning_rate = -50.0

#2、梯度下降循环
for epoch in range(epochs):
    xenc = F.one_hot(xs,num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1,keepdims=True)

    #损失计算负对数似然（NLL）
    correct_probs = probs[torch.arange(num_examples),ys]
    loss = -correct_probs.log().mean()

    #梯度清空
    W.grad = None

    loss.backward()

    W.data += learning_rate*W.grad

    if epoch%10 == 0:
        print(f"Epoch{epoch:3d} | Loss:{loss.item():.4f}")

print(f"Final Loss：{loss.item():.4f}")
```

- 特别注意：```correct_probs = probs[torch.arange(num_examples), ys]```它用两个并排的列表（一个是行号` 0, 1, 2...`，一个是正确答案的列号 `ys`），一瞬间就把所有二十多万个样本里，模型对“正确答案”的预测概率全部抠了出来！没有任何循环，全部在底层的 C++ 和 GPU 级别并行完成。
- 在这一步之前，`probs` 是一个巨大的二维矩阵（形状大概是 `[228146, 27]`）。每一行代表模型对当前样本预测出的 27 个字母的概率分布。
- 如果用原生 Python 找：
需要写一个循环：`for i in range(num_examples): prob = probs[i, ys[i]]`。这在深度学习里是灾难性的性能杀手。
- PyTorch 的高级索引（Fancy Indexing）：`torch.arange(num_examples)` 瞬间生成了一个巨大的行索引列表：`[0, 1, 2, ..., 228145]`。
`ys`是我们早就准备好的正确答案的列索引列表，比如 `[5, 13, 13, 1, 0...]`。
当把这两个列表同时塞进中括号里，PyTorch 底层的 C++ 引擎就会火力全开，像精确制导导弹一样执行配对抽取：“去第 0 行把第 5 列的数拿出来，去第 1 行把第 13 列的数拿出来……”。



## `Classification`&`Regression`
在构建神经网络时，损失函数的选择由目标任务的物理意义严格决定。前序在构建 MicroGrad 时采用了 MSE，而在当前的语言模型（Bigram）中使用了 NLL（负对数似然 / 交叉熵）。
- MSE (均方误差)：适用于**回归任务 (Regression)**。本质是衡量数值在空间上的绝对距离。
  适用于目标标签是连续的浮点数（如预测坐标、房价），数值之间存在明确的大小和远近关系的场景。
- NLL (负对数似然)：适用于**分类任务 (Classification)**。本质是衡量模型对正确答案的信心指数（概率重合度）。适用于目标标签是相互独立的平行类别（如 26 个字母、猫/狗）。类别之间地位平等，不存在空间上的“距离”概念。  
  
在字符预测任务中，真实标签被视为 `One-Hot`编码（即只有正确字母所在的索引位置概率应为 1.0，其余全为 0）
- 提取正确概率：`correct_probs = probs[torch.arange(num_examples), ys]`。由于真实标签极其纯粹，我们只需利用 PyTorch 的高级索引，直接把模型在“正确索引位置”上给出的预测概率 $p$ 抽取出来。
- 计算损失： loss = -correct_probs.log().mean()。我们不对预测概率做“减去 1.0”的距离计算，而是直接取负对数 $-\log(p)$。
  - 当 $p$ 越接近 1.0 时，损失趋近于 0（极其自信且正确）   
  - 当 $p$ 越接近 0.0 时，损失趋向于无穷大（严厉惩罚模型对正确答案的忽视）



## 实现最终预测
```python
# 设定和之前一模一样的随机种子，为了见证奇迹的时刻
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  out = []
  ix = 0 # 从特殊的起始符 '.' (索引 0) 开始
  
  while True:
    # --- 神经网络前向传播 ---
    # 1. One-Hot 编码
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    
    # 2. 穿过神经网络 (矩阵乘法)
    logits = xenc @ W 
    
    # 3. Softmax 层：把得分变成加起来等于 1 的概率分布
    counts = logits.exp()
    p = counts / counts.sum(1, keepdims=True) 
    
    # 根据神经网络算出来的概率 p，抽出下一个字母的索引
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    
    # 翻译回字母并记录
    out.append(itos[ix])
    
    # 如果抽到结束符 '.'，打破循环，完成当前这个名字的生成
    if ix == 0:
      break
      
  print(''.join(out))
```
```
cexze.
momasurailezityha.
konimittain.
llayn.
ka.
```

## 总结
对比前半部分用**查频次表（统计建模）**生成的那5个名字，会发现两种方法预测的一模一样
最核心的架构哲学：
- 用纯数学统计的办法，算出了一个完美的概率分布。
- 用一个单层的神经网络，随机初始化一个极其无知的权重 $W$。
- 用 NLL 损失函数和梯度下降，逼着神经网络去学习。
- 最终，神经网络学会的那个 $W$ 矩阵，在功能上完美等价于那个手工统计出来的频次矩阵

如果仔细观察生成出来的名字，虽然它们比完全随机的乱码要好一点，但依然很不通顺，根本不像真实的人名。
核心原因在于我们现在的模型叫做 `Bigram（二元语法）`。它的视野极其狭窄——它在预测下一个字母时，永远只能看见上一个字母，它完全没有记忆！
比如，在预测第 4 个字母时，它根本不知道第 1 个和第 2 个字母是什么。这种“只有一秒钟记忆的鱼”，是不可能写出复杂的人类语言的。