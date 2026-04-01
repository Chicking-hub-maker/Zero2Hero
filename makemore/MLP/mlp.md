# Multi-Layer Perceptron 多层感知机

## 前序准备
```python
import torch
import torch.nn.functional as F

words = open("names.txt",mode = 'r').read().splitlines()
chars = sorted(list(set('.'.join(words))))  #获取出现的字母，并排序
stoi = {s,i+1 for i,s in enumerate(chars)}  #构建字母到数字的映射
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

```

## 构建三维滑动窗口数据集

```python
block_size = 3
X,Y = [],[] #存储输入样本和目标输出

for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context += context[1:] + [ix]
  X = torch.tensor(X)  
  Y = torch.tensor(Y)  
```

```
-----基建完成-----
X shape:torch.Size([228146, 3])
Y shape:torch.Size([228146])
```

- 初始化上下文`context`：先全部用0填充，表示句子的起始
- 遍历每个单词并在末尾加上'.'，再转换为数字索引
- 将上下文内容索引添加进`X`,将需要预测的索引添加进`Y`
- `context`处理，去掉最左边的字符，加上当前字符的索引
- 开启下一个循环，继续滑动窗口

## 词嵌入`embedding`
在Part1中，我们为了让字母平等，使用了`One-Hot编码`，虽然带来了各个字母数学表示上的独立与平等，但是存在数学上的致命缺陷：    
- 在数学上，任何两个 One-Hot 向量之间的距离、角度都是完全一样的。在模型眼里，字母 `a` 和字母 `e` 之间的关系，就跟字母 `a` 和字母 `z` 之间的关系一样，毫无瓜葛，形同陌路.

- 但从实际人类语义的角度来看`a` 和 `e` 都是元音字母，它们在拼写名字时经常可以互相替换（比如 `Mary` 和 `Mery` 听起来都很合理，但 `Mzry` 就很违和）。因为 One-Hot 编码无法体现这种相似性，模型遇到没见过的组合时，就只能彻底瞎猜。

我们的做法
- 引入矩阵 $C$，把 27 个字母硬生生塞进了一个 2 维的连续空间（就像一张平面地图，有 X 轴和 Y 轴）。
- 一开始，因为 $C$ 是随机生成的（`torch.randn`），这 27 个字母在地图上是乱撒的。但是，随着接下来神经网络开始训练（计算 Loss 并反向传播），模型会为了把名字拼对，自己去移动这些字母在地图上的坐标！
- 模型如果在训练集里看到大量类似 `m-a-r-y, m-e-r-y, m-i-r-y `这样的名字，它就会在底层恍然大悟： a, e, i 这几个家伙在拼写时的作用是差不多的,于是，在经历几万次梯度下降后，`a, e, i, o, u` 这些元音字母，会在这个 2 维空间里自动聚拢成一团，成为好邻居。

### 核心意图：提升泛化能力
假设在你的几万个人名训练集里，从来没有出现过 `m-e-r-y` 这个词，只出现过 `m-a-r-y`。

- 如果是以前的 `Bigram` 模型： 遇到没见过的 `m-e-r-y`，直接懵逼，给出一个极低的概率。

- 现在的 `MLP` 模型： 当它看到 `m-e-r-y` 时，它会去地图上查 e 的坐标。它发现：e 的坐标离 a 非常近,既然 `m-a-r-y` 是个好名字，那 `m-e-r-y` 肯定也是个不错的好名字

**将离散的整数索引，变化为连续的浮点数向量，根本目的是想让神经网络能够衡量字母之间的语义相似度，从而获得强大的推理和泛化能力**


## 隐藏层`Hidden Layer`

```python
hidden_size = 100

#初始化隐藏层参数
W1 = torch.randn((6,hidden_size),generator=g)
b1 = torch.randn(hidden_size,generator=g)

#前向传播与.view()处理
h = torch.tanh(emb.view(-1,6) @ W1 + b1)

print(f"隐藏层输出h的形状:{h.shape}")
```

`.view(-1, 6)` 的底层艺术： 在计算机的最底层（比如 C/C++ 的内存管理级别），多维张量其实根本不存在，它们全都是在内存里排成一条直线的一维数组。`.view()` 极其优雅，它完全不移动或复制内存中的任何一个字节，它仅仅是修改了 PyTorch 读取这段连续内存的“步长（Stride）”和“元数据”。