# MicroGrad实现标量自动微分与API实现
(完整代码在文末)（`micrograd.py`为完整代码）
## 计算图的构建
在执行a*b、e+c的时候程序不仅是在进行数学上的运算，更是在搭建计算图。        
    
- `节点Node`：每一个Value实例都是图上的一个节点

- `边Edge`：`self._prev`记录了当前节点由哪些节点计算得来

- `局部梯度公式(_backward)`:每执行一次运算的时候，程序会立即定义一个闭包(Closure)`_backward`。并记录了当前操作的局部导数，并规定了如何将外部传来的梯度(`out.grad`)乘上局部导数后，传递给自己的节点(`self`和`other`)

```python
class Value:

    def __init__ (self,data,_children=(),_op='',label = ''):
        self.data = data 
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self,other):
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        
        out._backward = _backward
        return out

    def __mul__(self,other):
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def tanh(self):
            x = self.data
            t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
            out = Value(t,(self, ),'tanh')

            def _backward():
                self.grad += (1-t**2)*out.grad
            out._backward = _backward
            return out

```



## 拓扑结构的搭建
在反向传播的时候，我们面对一个重要问题：**必须先计算出某个节点所有下游节点的梯度，才能传播到自身，计算出L关于当前节点的梯度**，故而必须构建拓扑图才能够实现梯度计算
```python
 def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```
采用一个DFS，后序遍历计算图
- 沿着`_prev`一直往下找，直到没有前驱结点（来到最初的变量）,这里通过递归调用实现

- 当一个节点的所有输入节点都被`append`处理完成之后，这个节点才会被`append`到`topo`列表中

- 在生成的`topo`列表中，排在最前面的是最基础的输入变量如`a,b`，最后面的则是最终的输出变量`L`



## 反向传播的实现
有了拓扑排序好的节点序列，反向传播水到渠成
```python
self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```
- 设定起点：`self.grad=1.0`,此处的`self`指的是最终的结果节点`L`他对自己的导数 $\frac{\partial L}{\partial L}$ 始终是1

- 逆序执行：前面构建的`topo`列表是输入到输出的序列，翻转处理一下

- 链式法则传递：遍历到每一个节点的时候，都调用他身上绑定的`_backward()`函数



## 梯度累加机制：变量复用的处理`+=`
在计算图中如果一个变量(节点)被多条路径使用，那么在反向传播的时候，他的梯度必须是**所有依赖他的路径的偏导数之和**
假设有一个节点$x$,同时参与了两次不同的运算，分别生成了$y$和$z$，实际上最后是由$y$和$z$共同决定了最终的损失函数$L$根据多元微积分链式法则，$L$关于$x$的梯度公式应该为
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} + \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}
$$
这里的加号，体现在代码中就是+=，能够使得当计算完$y$关于$x$的梯度之后，在计算$z$关于$x$的梯度的时候，不会将之前的结果覆盖掉(如果只用`=`最终计算的只有$z$关于$x$的梯度)



## 闭包`Closure`

在自动引擎实现中，所依赖的一个很重要的功能就是闭包  
定义：是函数与其词法作用域中自由变量绑定的组合体。其本质是：内部函数即使在其定义的作用域销毁后，仍能访问并操作该作用域中变量的能力。    
 闭包 = 函数对象 + 其定义时词法环境中自由变量的持久化引用绑定（由运行时通过 `__closure__` 机制实现）,需要同时满足下面三个条件
  
- 嵌套函数定义：`_backward()`嵌套在`__add__()`中

- 引用外部作用域变量：在`_backward()`中使用了`self`,`other`,`out`这几个属于`__add__()`的参数

- 函数对象逃逸到外部作用域：`out._backward=_backward`,中对象通过`out`返回到了全局作用域

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    
    # 定义在 __add__ 内部的函数，这就是闭包！
    def _backward(): 
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
        
    out._backward = _backward # 将闭包绑定到输出节点上
    return out # __add__ 执行完毕，生命周期结束
```

### 为什么需要这么做
- 延迟执行：前向传播只负责搭建舞台排兵布阵（构建计算图），考虑函数执行的生命周期，我们将计算梯度这个动作、连同它涉及到的变量打包成一个闭包“挂”在节点上，这样就成为了这个节点实例的一个属性，后续再进行执行

- 状态捕获：当最后执行`node._backward()`时，虽然原来的加法运算早已结束，但是闭包能够将当时涉及到的变量值实现捕获留存`other`,`self`,再精准的传递给正确的父节点

- 函数对象留存：在Python中一切皆对象，`out._backward = _backward`将函数对象赋值给out这个Value实例的属性留存，后续在执行整个计算图的反向传播的时候，直接`node._backward()`调用计算属性


## 拓展：装饰器`Decorator`
打个通俗的比方：假设你写好了一个函数，它就像一个“原味甜甜圈”。现在你想给它加点料（比如计算执行时间、检查用户权限、打印日志），但你不想把甜甜圈切开（不想修改原函数的内部代码）。装饰器就是一台机器，它把你原来的函数丢进去，自动在外面裹上一层“巧克力糖霜”，然后还给你一个增强版的新函数。

### 核心痛点：不修改源码，如何增加功能
假设你有一个极其简单的函数：
```python
def say_hello():
    print("Hello World!");
```
老板提了个需求，在函数执行的前后必须打上分割线来记录日志，最笨的办法是直接改代码，加上 `print("---开始---") `和 `print("---结束---")`。但如果老板要求给100个函数都加上这个功能，你肯定会抓狂。

这里可以写一个“包装机器”函数，将需要包装的函数对象传进去
```python
def log_decorator(func):
    def wrapper():
        print("--- 准备执行 ---")
        func()  # 执行原来的原味函数
        print("--- 执行完毕 ---")

    # 把包装好的新函数（闭包）返回出去，记住不要加括号！
    return wrapper
```
此时就可以通过包装函数添加上这部分
```python
say_hello = log_decorator(say_hello)
say_hello()
```

### 终极形态：@语法糖
```python
@log_decorator
def say_hello():
    print("Hello World!")

say_hello() #此时调用的是功能已经增强了的函数

```
### 装饰器的本质
- 它是接收一个函数作为参数的函数
- 在其内部定义了一个闭包`Wrapper`，在闭包中添加了新功能并调用原函数
- 最终将这个闭包返回出去替换掉了原函数

## 功能拓展：构建完整运算库
### 自动类型转换
在之前的写法中，只能够处理`Value(2) + Value(3)`,当执行`a=Value(2.0) a+1`时将会报错
判断出`1`不是`Value`对象,故而需要额外处理
```python
other = other if isinstance(other,Value) else Value(other)
```

### 解决左侧数字问题
上一步解决了`a+1`的问题，但如果是`1+a`呢，解释器会先尝试调用整数1的内部方法1`__add__(a)`但普通的 int 类根本不认识你的 `Value`类，所以会报错并返回`NotImplemented`   
现在再采用备选方案去右边寻找`a`,看看它有没有定义 `__radd__` (Right Add)。你定义的 `__radd__` 直接 `return self + other`，就把`1+a`巧妙地转换回了`a+1` 从而顺利执行！

```python
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
```

这里实际用到的是Python中魔法方法的**降级机制**

### 补全运算算子
```python
def __pow__(self,other):
        assert isinstance(self,(int,float)),"only supporting int/float "
        out = Value(self.data ** other,(self,),f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
```

### 复用计算图实现复合运算
```python
   def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
```
在新增三个运算方法中，没有定义`_backward`闭包。     
例如在执行除法`a/b`的时候，底层实际上先调用`__pow__`生成一个中间节点$b^{-1}$,然后再调用`__mul__`    
实际上是通过利用基础算子搭建高级算子，而底层计算图和反向传播逻辑活自动通过基础算子的闭包链式拼接起来



## torchAPI等价实现
```python
import torch
x1 = torch.Tensor([2.0]).double()                
x1.requires_grad = True

x2 = torch.Tensor([0.0]).double()               
x2.requires_grad = True

w1 = torch.Tensor([-3.0]).double()               
w1.requires_grad = True

w2 = torch.Tensor([1.0]).double()                
w2.requires_grad = True

b = torch.Tensor([6.8813735870195432]).double()  
b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
```
### `Scalar`到`Tensor`
转换成高维度张量进行并行运算，实现加速

### 图构建的显式处理`requires_grad=True`
Pytorch中为了节省内存和提高性能，默认创建的张量是不记录梯度的（`required_grad=False`）所以需要显式的写明需要梯度记录（告诉底层这是一个需要通过梯度下降优化权重的参数），才会为其分配`.grad`属性和回传闭包（在Pytorch中叫做`grad_fn`）

### `.item()`的安全剥离
在Pytorch中如果直接打印或者引用`o`，此时不仅是提取了单个数据值，而是牵扯出来整个庞大的计算图，包含所有的历史父节点引用，并且全部混杂在内存中。  
`.item()`能够正确的剥离数值，极大程度避免了OOM

### 梯度累加问题`Gradient Accumulation`
前面花了很大功夫在手写引擎中`+=`的重要性，Pytorch底层完全遵循同样的链式法则，这就意味着，如果在一个循环中调用了多次`.backward()`每次算出的梯度一定会累加在`.grad`中 
在现实训练中每次执行新的`backward()`之前都要显式的将梯度清零（调用优化器的`optimizer.zero_grad()`来实现）

### `.backward()`的隐式起点
在这段代码中，`o` 是一个只包含单一元素的张量，所以 `o.backward()` 可以直接空手运行。它在底层等价于你的微型引擎中执行 `self.grad = 1.0` 这一步，用来点燃反向传播的火种。
但如果你的输出结果 `o` 是一个包含多个元素的向量（比如未经过汇总的 Loss 矩阵），直接调用 `o.backward()` 会直接报错。此时你必须传进去一个与 o 形状相同、由 1.0 组成的张量权重（例如 `o.backward(torch.ones_like(o))`），引擎才能知道该如何分配初始的梯度。



## 组合式架构设计（还原torch.nn模块）
整个架构采用搭积木的方式，将底层的数学运算层层封装，构建出宏观的网络结构(Neuron->Layer->MLP)

- ### 神经元`Neuron`
```python
import random
class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        out = torch.tanh(act)
        return out
    
    def parameters(self):
        return self.w + [self.b]
```
代码中构建了`Neuron`类，首先来看`__init__()`,需要输入`nin`表示输入的特征个数，在初始化时分别定义好了权重`w`和偏置`b`,需要注意的地方时这里的参数用`Value`类包裹起来，这就意味着这些参数纳入了自动微分的监控网路中；接着再看`__call_()`，实现了核心的数学公式
$$
y = \tanh(\sum_{i=1}^{n} w_i x_i + b)
$$
传入的`x`是一个列表`List`或者元组`tuple`，元素类型时封装好的`Value`类，函数中采用`zip()`函数，将多个可迭代对象（比如列表）里面对应的元素，像拉链的左右两边一样一对一的配对咬合在一起    
如果没有使用`zip()`，后续想要把`w`和`x`逐个相乘，就只能写传统的按索引遍历循环，较为复杂
```python
act = self.b
for i in range(len(x)):
    act += self.w[i] * x[i]
```
如果使用`zip()`，假设有`self.w = [w1,w2,w3]`和`x = [x1,x2,x3]`当执行`zip(self.w,x)`的时候，会自动打包成`[(w1,x1),(w2,x2),(w3,x3)]`紧接着就可以简洁的生成器表达式，完成整个的点积运算
```python
(wi*x1 for wi,xi in zip(self.w,x))
```

- ### 神经网络层`Layer`
```python
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] 
```
如果说一个神经元只能够从数据中提取一种特定的特征，那么一个Layer的本质就是将多个神经元并排的放在一起，让他们同时观察同一份数据从而多维度全方面的提取出各种不同的特征。类似的我们从三个和核心方法来解剖`Layer`
```python
def __init__(self,nin,nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
```
`nin`:这一层接收到的数据有几个维度，就传到相应神经元，让他们接受几个维度    
`nout`：由于每个神经元最后只输出一个数值，所以这一层神经元的个数就直接决定了这一层输出结果的维度大小


```python
def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
```
将同一份数据`x`喂给这一层所有的神经元，并将结果汇聚成一个分数列表`outs`

```python
def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
```
再将参数汇总成新的扁平化的参数表，注意这里的双重列表推导式子等价于两层循环
```python
params = []
for neuron in self.neurons:         
    for p in neuron.parameters(): 
        params.append(p)            
return params
```

- ### 多层感知机`MLP`
`MLP`将多个`Layer`首尾相连，很重要的一点就是每一层之间要实现维度吻合。  
同样我们从核心函数来看
```python
def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
```
`nin`  整个网络最开始接收的输入特征数  
`nouts`列表，包含每一层想要的神经元个数

假设要做一个二分类任务，输入特征有3个，中间包含两个隐藏层（每个隐藏层包含4个神经元），最后输出层一个神经元输出一个预测值，当我们实例化`MLP(3,[4,4,1])`的时候
- `sz = [3] + [4,4,1]`-->`sz = [3,4,4,1]` 此时`sz`构成了整个网络的设计图纸
- 紧接着列表推导式开始遍历`range(leg(nouts))`循环三次建立三个层Layer
  - `i=0`时：实例化`Layer(sz[0],sz[1])`即`Layer(3,4)`,表示第一层接收3个输入，输出4个结果
  - `i=1`时：实例化`Layer(sz[1],sz[2])`即`Layer(4,4)`,表示第二层接收上一层传来的4个输入，输出4个结果
  - `i=2`时：实例化`Layer(sz[2],sz[3])`即`Layer(4,1)`,表示第三层接收4个输入，输出1个结果 

如此，便实现了完美的各个维度的吻合,接下来通过`__call__()`来实现数据流通


```python
def __call__(self,x):
    for layer in self.layers:
        x = layer(x)
    return x
```
数据随神经网络不断向前流动，每经过一层`Layer`上层的输出结果就直接覆盖原来的变量x，成为新的输出送给下一层    
当你执行 `layer(x)` 时，它底层调用了本层的 `[n(x) for n in self.neurons]`；而`n(x)` 底层又调用了 `Value` 类的加法、乘法和激活函数。一次 `MLP` 的前向传播，会在内存中瞬间生成包含成百上千个 `Value` 节点的庞大计算图（DAG）。

最终在进行终极的参数大一统
```python
def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

### 架构复盘：**组合模式**
从微观的 `Value`，到 `Neuron`，再到 `Layer`，最后到宏观的 `MLP`，它们对外暴露的接口是高度统一的。根本不需要关心内部实现，只要给它数据，它就能前向计算；只要跟它要参数，它就能打包交出


## 完整代码
```python
import math
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```