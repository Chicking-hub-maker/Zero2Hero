# 反向传播深入`Backward Propagation`
在之前的学习中，我们从底层用Python从底层构建了能处理标量求导的微型引擎Micrograd，后续的Bigram、MLP，使用的一直是`.backward()`带来的便利，现在需要做的是打开这个黑盒，从底层求导手工实现每一个张量的反向传播

## 两个基本直觉的建立与理解
### 1、广播与求和的“互逆”关系
我们讨论过“梯度累加机制”：如果一个节点 $x$ 参与了多次运算生成了 $y$ 和 $z$，那么在反向传播时，必须根据多元链式法则，把所有依赖它的路径的偏导数加起来：$dx = dy \cdot \frac{\partial y}{\partial x} + dz \cdot \frac{\partial z}{\partial x}$。在张量操作中，最隐蔽的“节点复用”就是广播机制（Broadcasting）。    
假设我们有一个列向量 $X$ (形状为 3x1) 和一个行向量 $Y$ (形状为 1x3)。
$$X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \quad Y = \begin{bmatrix} y_1 & y_2 & y_3 \end{bmatrix}$$
执行加法 $Z = X + Y$ 时，PyTorch 并不会在内存里真的把 $X$ 复制 3 遍变成 3x3。在底层系统级别，它只是巧妙地把张量的 Stride（步长）设为了 0 。这意味着在进行加法计算时，内存指针在遇到某个维度时原地踏步，从而虚拟出了一个 3x3 的矩阵 $Z$：
$$Z = \begin{bmatrix} x_1+y_1 & x_1+y_2 & x_1+y_3 \\ x_2+y_1 & x_2+y_2 & x_2+y_3 \\ x_3+y_1 & x_3+y_2 & x_3+y_3 \end{bmatrix}$$
现在看反向传播：假设我们最终有一个标量损失 $L$，并且我们已经拿到了 $Z$ 的梯度矩阵 $dZ$（形状也是 3x3）。
$$dZ = \begin{bmatrix} dz_{11} & dz_{12} & dz_{13} \\ dz_{21} & dz_{22} & dz_{23} \\ dz_{31} & dz_{32} & dz_{33} \end{bmatrix}$$
我们想求 $X$ 的梯度 $dX$。目光聚焦到 $X$ 的第一个元素 $x_1$。在正向传播中，$x_1$ 被用了3次，它参与生成了 $Z$ 的第一行：$Z_{11}, Z_{12}, Z_{13}$。根据链式法则，任何用了 $x_1$ 的地方，都会把梯度反传给它。因此，$x_1$ 的梯度就是它产生的这三个子节点的梯度之和：
$$dx_1 = dz_{11} \cdot 1 + dz_{12} \cdot 1 + dz_{13} \cdot 1$$
推而广之，$dX$ 的每一行，就是把 $dZ$ 对应行的数据全部加起来。在代码层面，这非常直观：因为 $Y$ 在正向传播时被广播了第 0 维（行），所以 $dY$ 就是 dZ.sum(0)；$X$ 被广播了第 1 维（列），所以 $dX$ 就是 dZ.sum(1, keepdim=True)。

### 2、矩阵乘法构建形状匹配的转置直觉
面对 $Y = X \cdot W$，传统的微积分课程会教你写出复杂的雅可比矩阵（Jacobian）。但在深度学习工程中，我们要抛弃这种做法，用纯粹的形状匹配和物理意义来倒推。假设：$X$ 是输入数据（比如 Batch Size 是 $N$，特征数是 $D$），形状为 $(N, D)$。$W$ 是权重矩阵（输入 $D$，输出 $M$），形状为 $(D, M)$。$Y$ 是输出，形状为 $(N, M)$。反向传播时，我们手里已经有了来自上游的梯度 $dY$，它的形状必然与 $Y$ 一模一样，即 $(N, M)$。我们的目标是求 $dW$ 和 $dX$。
$$dW = X^T \cdot dY$$
$$dX = dY \cdot W^T$$

## Level 1 拆解交叉熵
在之前，我们计算 Loss 只需要一句极其方便的代码：
```python
loss = F.cross_entropy(logits,Yb)
```
这是`融合算子Fused Kernel`，是Pytorch在底层用C++/CUDA将复杂运算一次性打包做完，并能够给出最终梯度。但这里需要将其拆解成为基本的加减乘除运算，唯此我们能够将梯度通过链式法则传递回去。  

假设我们已经算出了最后一层的输出`logits`（未归一化的得分，形状是`[32, 27]`，即 32 个样本，27 个字符）

```python
#1、寻找每一行的最大值
logit_maxes = logits.max(1,keepdim=True).value

#2、减去最大值
norm_logits = logits - logit_maxes

#3、求指数e^x，将得分化为正数
counts = norm_logits.exp()

#4、求和，得到分母
counts_sum = counts.sum(1,keepdim=True)

#5、取倒数
counts_sum_inv = counts_sum**-1

#6、计算最终概率softmax
probs = counts *counts_sum_inv

#7、求自然对数
logprobs = probs.log()

#8、抽出正确答案对饮的log概率，求均值并取负号NNL
loss = -logprobs[range(n),Yb].mean()
```
### 步骤拆解的基本说明
- 第一步减去最大值，是个经典的数值稳定技巧：如果某个 `logits` 的值是 100，`exp(100)` 会得到一个极其巨大的数字（$2.68 \times 10^{43}$），计算机的浮点数会直接溢出（变成 `NaN` 或 `Inf`）。但如果我们找出最大的那个数，让所有的数都减去它。那么最大的数变成了 0，`exp(0) = 1`。其他的数都是负数，exp(负数) 是介于 0 到 1 之间的小数。这样既不会溢出，又完全不影响 `Softmax` 的最终概率结果（因为分子分母约掉了同一个常数 $e^{max}$）
- 为什么不直接除以 `counts_sum`，而是要写成 `**-1` 然后相乘（第 5、6 步）?如果写 `probs = counts / counts_su`m，数学上完全没问题。但是，**除法在求导时涉及复杂的商法则**。如果写成 `counts * (counts_sum 的 -1 次方)`，在后面写`Backward`时，就可以完全复用乘法法则和幂函数法则。此外这么做是为了保证我们手算的梯度，和 PyTorch 底层算出来的梯度能够做到 bit-exact（比特级的完全一致）。哪怕是浮点数精度的微小误差，在这个严苛的测试中都不被允许！
- 第 八 步的魔法索引 `[range(n), Yb]`，`n 是 batch_size（这里是 32）`。`Yb` 是包含 32 个正确答案索引的张量。这句话的意思是：“在第 0 行，拔出第 `Yb[0]` 列的值；在第 1 行，拔出第 `Yb[1]` 列的值...”。这就是交叉熵的核心：我只关心你把‘正确答案’预测到了多少概率，至于你给错误答案分了多少，我通过前面 `Softmax` 的分母已经综合考虑过了。

### 反向推导
完成拆解后，从最终的结果-标量`loss`开始，讲梯度逐步反推回最初的logits

- `step8:loss->logprobs`
  - 前向：`loss = -logprobs[range(n),Yb].mean()`使用高级索引，直接按照正确答案找出正确答案对应的分类的那个概率值.`logprobs`形状为`[32,27]`
  - 均值操作本质是除以`n`,故而局部导数即`1/n`,加上前面的符号，故这步`-1/n`
  - 其余元素没有参与计算，对梯度影响为0

- `step7:logprobs->probs`
  - 前向：`logprobs = probs.log()`纯粹元素级操作，形状没有发生改变`[32,27]`
  - 自然对数操作，求导局部导数为`1/x`

- `step6:probs->counts,counts_sum_inv`
  - 前向：`probs = counts*counts_sum_inv` 形状`[32,27] = [32,27]*[32,27]`这里的`counts_sum_inv`本来是`[32,1]`满足第二维度是1，自动触发广播机制

- `step5:counts_sum_inv->counts_sum`
  - 前向：`counts_sum_inv = counts_sum**-1`元素级操作，形状为`[32,1]`
  - 取了倒数，按照法则，局部导数为`-x**-2`

- `step4:counts_sum->counts`
  - 前向：`counts_sum = counts.sum(1,keepdim=True)`通过求和计算讲`[32,27]`拍扁成`[32,1]`
  - 反向：“求和的逆运算就是广播”。正向传播时，每一行的 27 个数共同加起来得到了一个总和。那么在反向传播时，这个总和的梯度`dcounts_sum`应该一视同仁地**原样复制(广播)** 给这 27 个元素。
  - 此外还需要注意这里需要使用`+=`进行梯度累加，因为counts参与了多次运算

- `step3:counts->norm_logits`
  - 前向：`counts = norm_logits.exp()`元素级操作，形状 `[32,27]`
  - `exp()`导数不变，直接复用前向传播的结果

- `step2:norm_logits = logits,logit_maxes`
  - 前向：`norm_logits = logits - logits_max`其中`logits[32,27]`,`logits_maxes[32,1]`,运算的时候进行了广播
  - 反向处理的时候就要对应求和

- `step1:logits_max->logits` 

  - 前向：`logits_max = logits.max(1,keepdim=True).values`找出每一行的最大值`[32,27]->[32,1]`
  - max 操作是一个典型的路由（Routing）操作。就像 Step 8 一样，谁在正向传播中被选为了最大值，谁就承担反向传播的梯度责任。那些没被选上的“失败者”，梯度为 0。
  - 而且，这也是 logits 第二次出现，所以我们要用 += 累加梯度。
  - 我们要找出每一行最大值所在的位置（logits.max(1).indices），把它们变成 One-Hot 掩码，然后把 dlogit_maxes 分发进去。

- 回顾一下，会发现看似狂野复杂的矩阵求导，被拆解后全都是基本的局部微积分法则 + 形状匹配 + 广播与求和的镜像对称 + 梯度累加。这就是现代深度学习框架底层的物理真相。 


## `Softmax`与`NLL`的联合
上一部分通过从最终的`loss`一步步的反向求出`dlogits`，但是当将Softmax与NLL连起来求导会发现数学表达很简洁
$$\frac{\partial L}{\partial logits_i} = p_i - y_i$$
其中 $p_i$ 是 Softmax 算出的概率，$y_i$ 是真实标签（正确答案位置为 1，其余为 0）。
物理意义（极度硬核）：假设正确答案是 'a'，模型预测 'a' 的概率是 $0.1$。那么梯度就是 $0.1 - 1.0 = -0.9$。这是一个极强的负梯度，疯狂地把 'a' 对应的 logit 往上拉！假设另一个错误答案 'b'，模型预测的概率是 $0.2$。那么梯度就是 $0.2 - 0 = +0.2$。这是一个正梯度，像地心引力一样把 'b' 的 logit 往下拉！如果模型完美预测了正确答案（概率接近 1.0），梯度就是 $1.0 - 1.0 = 0$，模型满意地停止更新：“不用拉了，已经很完美了。”
```python
dlogits = F.softmax(logits,1)#前向使用softmax先拿到概率
dlogits[range[n],Yb] -= 1#减去正确位置的概率（1或零）
dlogits /= n
```
短短 3 行代码，完美等价于之前繁琐的 8 步操作！这就是为什么现代框架中 F.cross_entropy 被称为 Fused Kernel（融合算子），它在底层 C++/CUDA 中直接用这个数学捷径算梯度，速度极快且极其省内存


## Level 2 矩阵运算与Tanh饱和问题
结合上一步求出的关于交叉熵的反向传播简洁处理，这一步可以直接反向穿透运算，推导出对应的梯度
### 第二个线性层
```python
logits = h @ W2 + b2
```
已知内容

- 上游回传的梯度`logits[32,27]`   
- 前向隐藏的状态`h[32,64]`    
- 第二层权重`W2[64,27]`
- 第二层偏置`b2[27]`

#### 求`db2`广播->求和
前向传播中，`b2`是`[27]`的一维向量，在和`h@W2`结果相加的时候，在`第0维`被广播了32次
```python
db2 = dlogits.sum(0)
```
#### 求`dW2`矩阵乘法->形状匹配
根据之前的形状匹配转置直觉
```python
dW2 = h.T @ dlogits
```
#### 求`dh`
同理
```python
dh = dlogits @ W2.T
```

### 非线性激活层
```python
h = torch.tanh(hpreact)
```
已知内容
- hpreact 是过激活函数前的值，形状 [32, 64]。
- h 是激活后的值，形状[32, 64]。
- 刚才求出的 dh [32, 64]。

$\tanh(x)$ 的导数是 $1 - \tanh^2(x)$。
前向传播里的 h 本身就是 $\tanh(x)$ 的结果.所以局部导数极其简单：直接就是 $1 - h^2$。根据链式法则，把上游传下来的 dh 和局部导数逐元素（Element-wise）相乘：
```python
dhpreact = (1.0 - h**2) * dh
```

### Tanh饱和问题
这个局部导数 $(1 - h^2)$。如果前向传播时，神经元极其兴奋或极其被动（h 接近 $1$ 或 $-1$），那么 $h^2 \approx 1$，局部导数 $(1 - h^2)$ 就会变成 $0$。这时候，不管上游的 dh 有多大，乘上 $0$ 之后，传给底层的 dhpreact 全是 $0$！这就是经典的**“梯度消失（Gradient Vanishing）”或“神经元坏死”**。  
也就是在之前笔记提到的，为什么要引入 Kaiming 初始化 和 BatchNorm 的根本原因。接下来，将进行整个网络中最复杂的Batch Normalization 层的反向传播


## Level 3 Batch Norm

```python
#1、求均值
bnmeani = 1/n*hprebn.sum(0,keepdim=True)

#2、求偏差
bndiff = hprebn-bnmeani

#3、偏差平方
bndiff2 = bndiff**2

#4、求方差
bnvar = 1/(n-1)*(bndiff2).sum(0,keepdim=True)

#5、标准差倒数
bnvar_inv = (bnvar+1e-5)**-0.5

#6、归一化
bnraw = bndiff*bnvar_inv

#7、放缩平移
hpreact = bngain*bnraw + bnbias
```
### 反向推导
- `step7:hpreact = bngain * bnraw + bnbias`
  - $bngain$ 和 $bnbias$ 都是可学习的参数，形状是 `[1, 64]`,发生广播
  - ```python
    dbnbias = dhpreact.sum(0,keepdim=True)
    dbngain = (dhpreact*bnraw).sum(0,keepdim=True)
    dbnraw = dhpreact*bngain
    ```
- `step6:bnraw = bndiff * bnvar_inv`   
  - ```python
    dbnvar_inv = (dbnraw*bndiff).sum(0,keepdim=True)
    dbndiff = dbnraw*bnvar_inv
    ```
- `step5:bnvar_inv = (bnvar + 1e-5)**-0.5`
  - 纯粹的幂法则 $f(x) = x^{-0.5} \rightarrow f'(x) = -0.5x^{-1.5}$。
  - ```python
    dbnvar = (-0.5*(bnvar+1e-5)**-1.5)*dbnvar_inv
    ```
- `step4:bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)`
  - 正向时，我们把 bndiff2 的 32 行数据压扁成了一行。反向时，我们要把 dbnvar 广播回 32 行，并乘上系数 $\frac{1}{n-1}$
  - ```python
    dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
    ```
- `step3:bndiff2 = bndiff**2`
  - ```python
    # 乘上局部导数 2*bndiff，并累加到之前的 dbndiff 上
    dbndiff += (2 * bndiff) * dbndiff2
    ```

- `step2:bndiff = hprebn - bnmeani`
  - ```python
    # hprebn 拿到第一部分梯度
    dhprebn = dbndiff.clone()

    # bnmeani 顺着广播维度求和收回
    dbnmeani = (-dbndiff).sum(0)
    ```
- `step1:bnmeani = 1/n*hprebn.sum(0, keepdim=True)`
  - ```python
    # 将均值的梯度广播回整个矩阵，并累加给 dhprebn
    dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
    ```

### 简化推导
```python
# 直接一行搞定整个 BatchNorm 的反向传播！
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact 
    - dhpreact.sum(0) 
    - n / (n-1) * bnraw * (dhpreact * bnraw).sum(0)
)
```
## Level 4 整合推导
