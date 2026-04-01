# `Batch Norm`批量归一化

## 现有问题
- 网络刚初始化还没开始训练时，`Loss`异常高。这是因为网络在“瞎猜”且非常自信`Softmax confidently wrong`
- 隐藏层的非线性激活函数`Tanh`大量输出 `-1 或 1`，导致`局部梯度为 0`（神经元“坏死”，Tanh 饱和）
- 考虑引入深度学习中著名的 `Kaiming 初始化 (He Initialization)`，通过缩放系数来控制每一层参数的初始大小，防止前向传播时数据分布爆炸或缩小。
- 即使有 `Kaiming 初始化`，对于极深的网络仍然很脆弱。既然我们希望进入 Tanh 之前的数据分布呈现标准正态分布（均值为 0，方差为 1），那么就可以直接在代码里强制把它标准化呢？这就是**批量归一化Batch Normalization** 的发明动机

## 前序准备
```python
words = open("names.txt",mode='r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
```

## 数据集构建与划分
```python
block_size = 3

def build_dataset(words):
    X,Y = [],[]

    for w in words:
        context = [0]*block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]    #滑动窗口构建，更新上下文
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"X.shape{X.shape}")
    print(f"Y.shape{Y.shape}")
    return X,Y

random.seed(42)
random.shuffle(words)

#划分数据集
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,Ytr = build_dataset(words[:n1])
Xval,Yval = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:])
```

## BatchNorm层配置
```python
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size,n_embd))
W1 = torch.randn((n_embd*block_size,n_hidden),generator=g)*(5/3)/((n_embd*block_size))**0.5
#b1 = torch.randn(n_embd*block_size,n_hidden,generator=g)
W2 = torch.randn((n_hidden,vocab_size),generator=g) * 0.01
b2 = torch.randn(n_hidden,vocab_size,generator=g) * 0

#BatchNorm Parameters
bngain = torch.ones((1,n_hidden))
bnbias = torch.zeros((1,n_hidden))
bnmean_running = torch.ones((1,n_hidden))
bnstd_running = torch.zeros((1,n_hidden))

parameters = [C,W1,W2,b2,bngain,bnbias]

print(f"The total num of parameters:{sum(p.nelement() for p in parameters)}")

for p in parameters:
    p.required_grad = True
```

- `Kaiming 初始化`的标准公式： W1 后面乘的那一长串 `(5/3) / ((n_embd * block_size)**0.5)`
  - `(n_embd * block_size)`是这一层的输入神经元数量（也叫 `fan_in`，这里是 10 * 3 = 30）。除以 $\sqrt{\text{fan\_in}}$ 是为了保证经过矩阵乘法后，方差不会被放大。
  - `5/3` 是针对 Tanh 激活函数的`增益系数 (Gain)`。因为 Tanh 会把数据向原点压缩，使得方差变小，所以需要乘以一个大于 1 的数把它“拉”回来一点。

- 关于`b1`被注释掉：因为在 `W1 @ X` 之后，我们马上要接 `Batch Normalization层`第一步就是减去这一个 `batch`的均值.如果加上了偏置 b1，然后立马去算均值并减去均值，这个`b1`的效果会被完全抵消掉。保留它只会白白浪费内存和计算资源，还会产生无用的梯度。

```python
#BatchNorm Parameters
bngain = torch.ones((1,n_hidden))
bnbias = torch.zeros((1,n_hidden))
bnmean_running = torch.ones((1,n_hidden))
bnstd_running = torch.zeros((1,n_hidden))
```
四个BN的重要参数
- 可学习的参数`bngain`和`bnbias`:即BatchNorm中的$\alpha,\beta$ 初始化为1和零。
  - 为什么`bngain`初始是 1，`bnbias`初始是 0？因为我们希望在训练刚开始的时候，BatchNorm 层只做纯粹的标准化（均值变0，方差变1），不改变数据的分布，这样数据能完美适配后面的 Tanh 函数。但在后续训练中，网络可以通过反向传播修改这两个值，自己决定要不要把分布再平移或缩放一点。
- 全局滑动平均的参数`bnmean_running`和`bnstd_running`:这两个参数是为解决推理时的问题:
    - 训练时，我们能算出每个 batch 的均值和方差。但是训练完，拿单个名字去测试/推理时，只有一个样本，算不了均值和方差？
    - 解决办法就是在训练过程中，像“记账”一样，每次算出一个 `batch` 的均值/方差，就顺手更新到这个 `running` 变量里（通常是旧值占 `99.9%`，新 batch 占 `0.1%`）。测试的时候，直接拿这两个全局统计好的值作为参数使用

## 传播进行
```python
max_step = 20000
batch_size = 32

lossi = []

for i in range(max_step):
    #MiniBatch
    ix = torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb = Xtr[ix],Ytr[ix]

    emb = C[Xb]
    embcat = emb.view(emb.shape[0],-1)  #构建好（32，30）的、词嵌入处理好的向量
    #对应W1 = torch.randn((n_embd*block_size,n_hidden),generator=g)

    hpreact = embcat @ W1

    #------------------------------------------
    #BatchNorm Layer

    bnmeani = hpreact.mean(0,keepdim=True)
    bnstdi = hpreact.std(0,keepdim=True)

    #Normalization and ScalingShifting
    hpreact = (hpreact-bnmeani) / bnstdi
    hpreact = hpreact*bngain + bnbias


    with torch.no_grad():
        bnmean_running = 0.999*bnmean_running + 0.001*bnmeani
        bnstd_running = 0.999*bnstd_running + 0.001*bnstdi


    #Non-Linear
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Yb)
```
- 注意`with···`部分，目的是为了在训练过程中，悄悄记录整个数据集的全局均值和方差，以供推理的时候使用。前面在定义Batch Norm参数的时候也提到过
- 关于`with torch.no_grad()`，表明下面两行只是更新统计量，不是模型计算图的一部分
  - 不需要反向传播
  - 不需要计算梯度
  - 实现内存节约和速度加快


## 封装构建类
如果将神经网络层数加深，再用散装的全局变量，分散的构建网络层和BatchNorm则不现实，故而我们需要构建封装好的核心组件，提升工程性
```python
#构建封装类，实现核心组件

#--------------线性层----------------
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight = torch.randn((fan_in,fan_out),generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

#--------------批次归一化层----------------
class BatchNorm1d:
    def __init__(self,dim,eps = 1e-5,momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        #可学习的参数
        self.gamma = torch.ones(dim)    #bngain
        self.beta = torch.zeros(dim)    #bnbias
        #全局平滑统计量
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self,x):
        #需要做分流逻辑处理，判断当前是训练还是推理，采用不同的计算方式
        if self.training:
            xmean = x.mean(0,keepdim=True)
            xvar = x.var(0,keepdim= True)
        else:
            xmean = x.running_mean
            xvar = x.running_var
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out
    
    def parameters(self):
        return [self.gamma,self.beta]
    
#--------------非线性激活层----------------
class Tanh:
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
```

