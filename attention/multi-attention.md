# `multi-attention`多头注意力的实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        #最后的线性投影层
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self,x):
        #1、并行执行所有注意力头
        head_outputs = [h(x) for h in self.heads]

        #2、在特征维度上进行拼接
        out = torch.cat(head_outputs,dim = -1)

        #3、经过一次线性投影，混合特征
        out = self.proj(out)

        return out
```        

- 假设我们的总特征维度 `n_embd = 32`。我们想要开启 `num_heads = 4` 个注意力头。那么分配给每个头的 `head_size` 就是 `32 / 4 = 8`。第一步：并行执行与拼接 `(torch.cat)`每个 `Head` 在内部生成自己的 $W_q, W_k, W_v$ 矩阵，独立计算自己的注意力分数，并最终吐出一个形状为 `[B, T, 8]` 的张量。因为有 4 个头，我们得到了 4 个这样的张量。当我们在 `dim=-1`（即特征维度）上执行 `torch.cat` 时，这 4 个大小为 8 的特征块被严丝合缝地拼接在一起。8 + 8 + 8 + 8 = 32。拼接后的 out 张量，形状完美恢复成了` [B, T, 32]`，即` [B, T, n_embd]`。
- 此处注意，还需要进行一次线性投影进行特征混合：在执行完`out = torch.cat(head_outputs, dim=-1)` 之后，我们的张量形状确实完美恢复成了 `[B, T, n_embd]（比如 [B, T, 32]）`。
但是，如果在内存里把这 32 个维度切开看，它的物理排列是极其生硬的：
`[头1的8维] + [头2的8维] + [头3的8维] + [头4的8维]`这 4 个头在刚才的注意力计算中是绝对并行且互相物理隔离的。头 1 根本不知道头 2 算出了什么。如果我们就把这种“拼盘”直接扔给下一层神经网络，下一层在读取数据时，面对的其实是 4 个割裂的特征孤岛。
- 为了打破这种物理隔离，我们让拼接后的张量穿过 `self.proj = nn.Linear(n_embd, n_embd)`。在底层，这相当于生成了一个形状为 `[n_embd, n_embd]（即 [32, 32]）`的权重矩阵 $W_O$。执行的矩阵乘法形状变化：`[B, T, 32] @ [32, 32] -> [B, T, 32]`。
$$Output = Concat\_Out \cdot W_O$$  
- 如何理解特征融合这一步：参数$W_o$可训练，得到对于各个特征融合的权重分布
  - 拼接前： 头 1 发现“这个词是主语”，头 2 发现“这个词有消极情感”。它们各说各的。
  - 投影后： $W_O$ 矩阵通过梯度下降学会了一种“联合翻译”的规则。它把头 1 和头 2 的信息融合在一起，生成了一个更高级的综合语义：“这是一个带有消极情感的主语”。


## 前馈神经网络`Feed Forward`
```python
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super.__init__()

        # 按照 Transformer 的标准架构，隐藏层的维度通常会放大 4 倍
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd)#最后的投影层
        )

    def forward(self,x):
        return self.net(x)
```

## 宏观架构`Transformer Block`
```python
class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super.__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head,head_size=head_size)
        self.ffwd = FeedForward(n_embd)

        #Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        # 这里的 x = x + ... 就是残差连接 (Residual Connection)
        # 注意：现代 Transformer (如 GPT-2) 普遍采用 Pre-Norm 架构，
        # 即在进入 Attention 和 FeedForward 之前先做 LayerNorm。

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```
### 残差连接
这就相当于在计算图中修建了一条从顶层直通底层的梯度高速公路。不管网络多深，最原始的信息和最纯粹的梯度都能畅通无阻地流转。

### 层归一化`LayerNorm`
在 BatchNorm 里，我们是沿着 Batch 维度去求均值和方差的（把 32 个样本的同一个特征压扁）。但在语言模型中，由于每个句子的长度 $T$ 是不断变化的，BatchNorm 的表现会极其糟糕。

`LayerNorm` 改变了归一化的方向。它不在样本之间（Batch）做统计，而是在单个样本的特征维度`（Channels）`上做统计。对于 x 形状为 `[B, T, C]` 的张量：
`LayerNorm` 会沿着 C 维度去求均值和方差。也就是把每一个词向量自己内部的 $C$ 个特征值变成均值为 0、方差为 1 的正态分布。这种设计让它完全脱离了对 `Batch Size` 和序列长度的依赖，计算极其稳定，是 Transformer 的不二之选。

### `Pre-Norm`架构
在原始的`《Attention Is All You Need》`论文中，`LayerNorm` 是放在加法之后的（即 `x = self.ln(x + sa(x))`），这叫 `Post-Norm`。但我们在上面代码里写的是 `x = x + self.sa(self.ln(x))，这叫 Pre-Norm。`
这也是 GPT 系列在工程上的一个极其重要的改进：把归一化放在分支计算之前，把高速公路彻底让给残差连接。 实验证明，这种微调能让极深的网络在训练初期稳定得多，甚至不再需要复杂的学习率热身（Warm-up）。