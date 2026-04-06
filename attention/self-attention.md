# `self-attention`自注意力机制的实现
## 前序内容
```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------
# 1. 全局超参数定义 (在真实架构中，这些通常写进 Config 类)
# ----------------------------------------------------
batch_size = 32    # B: 并行处理的序列数量
block_size = 8     # T: 序列的最大上下文长度 
n_embd = 32        # C: 每一个 token 被映射到的特征维度 

head_size = 16     # 单个注意力头投影后的特征维度 (通常是 n_embd // num_heads)

torch.manual_seed(1337) # 保证每次输出一样，方便调试

# ----------------------------------------------------
# 2. 模拟底层的输入数据流动
# ----------------------------------------------------
# 假设这是已经经过词嵌入层 (Token Embedding) 的数据
# 形状: [B, T, C] = [32, 8, 32]
x = torch.randn(batch_size, block_size, n_embd)
```
- 这里先忽略跳过词嵌入过程，直接开始处理输入
- 输入x的形状为[B,T,c]

## 注意力头构建
```python
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)

      
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape

        k = self.key(x) 
        q = self.query(x)

        #计算注意力分数（包含缩放点积注意力）
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5#输出矩阵[T,T]

        #因果掩码
        # 动态切片截取：只从预先分配好的大矩阵 self.tril 中，切出当前句子实际长度的 [:T, :T] 块
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))

        #归一化为概率分布
        wei = F.softmax(wei,dim=-1)#形状[B,T,T]

        #信息聚合
        v = self.value(x)
        out = wei @ v

        return out
```

这里有很多代码细节需要注意
- `__init__()`中定义了`p,k,v`三个权重矩阵即$W_k,W_q,W_v$形状都是`[C,head_size]`
- 后续在执行`k = self.key(x)`的时候，底层实际进行了高维度的矩阵乘法
  - `x`的形状`[B,T,C]`，$W_k$形状`[C,head_size]`
  - 这里维度不对应，实际是三维*二维。PyTorch 的 `nn.Linear` 被设计为只对输入张量的最后一个维度（特征维度）进行线性映射，而把前面所有的维度都视作互相平行的“样本”
    - 压扁（Flatten）： 它在逻辑上把前两个维度融合，将输入张量看作是形状为 `[B * T, C]` 的巨大二维矩阵。这就相当于把 $B$ 句话、每句话 $T$ 个字，全部拆散成 $B \times T$ 个互相独立的词向量，排成一条长长的队伍。
    - 二维矩阵乘法： 执行常规的矩阵乘法：`[B * T, C] @ [C, head_size]`，得到的结果形状是 `[B * T, head_size]`。
    - 折叠还原（Reshape）： 乘完之后，它再按照原来的批次和时间步长，把队伍重新折叠回去，输出最终形状 `[B, T, head_size]`。
- 在计算注意力分数的时候`q,k`的形状都是`[B,T,head_size]``.transpose(-2,-1)`将后两个维度交换，即公式中对应的转置
- 在 PyTorch 中，当两个三维张量使用 @ 相乘时，它会触发批量矩阵乘法（Batched Matrix Multiplication）它的规则是：死死锁定第一个维度 B，然后在内部对后面的二维矩阵进行标准的矩阵乘法。在底层的 C++/CUDA 代码里，你可以把它想象成存在这样一个隐形的循环（虽然实际上是极致并行的）：
  - ```python
    out = torch.zeros(B, T, T)
    for b in range(B): # 遍历每一个独立的句子 (Batch)
    # 取出当前句子的 q 矩阵: 形状 [T, head_size]
    q_b = q[b] 
    
    # 取出当前句子的 k 转置矩阵: 形状 [head_size, T]
    k_T_b = k.transpose(-2, -1)[b]
    
    # 执行 2D 矩阵乘法
    # [T, head_size] @ [head_size, T] -> [T, T]
    out[b] = q_b @ k_T_b
    ```
  - 得到`[T,T]`矩阵，在加上外层的`B`维度,最终输出的`wei`张量的形状就是`[B,T,T]`

## 因果掩码(Causal Masking)
在构建好的`wei`矩阵中`[B,T,T]`是各个数值Q和K的计算结果，但是肯定会存在非法数据（预知未来的计算数值），就需要使用下三角矩阵`tril`进行掩码处理（在__init__()中预留好了）
```python
# 将右上角（属于未来的部分，即 tril == 0 的区域）全部替换为负无穷大
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

## 信息聚合(Value Aggregation)
```python
v = self.value(x) # 形状: [B, T, head_size]
out = wei @ v     # 形状: [B, T, T] @ [B, T, head_size] -> [B, T, head_size]
```