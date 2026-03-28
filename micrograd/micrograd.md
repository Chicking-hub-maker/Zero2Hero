# MicroGrad实现标量自动微分

## 计算图的构建
在执行a*b、e+c的时候程序不仅是在进行数学上的运算，更是在搭建计算图。        
    
- `节点Node`：每一个Value实例都是图上的一个节点

- `边Edge`：`self._prev`记录了当前节点由哪些节点计算得来

- `局部梯度公式(_backward)`:每执行一次运算的时候，程序回立即定义一个闭包(Closure)`_backward`。并记录了当前操作的局部导数，并规定了如何将外部传来的梯度(`out.grad`)乘上局部导数后，传递给自己的节点(`self`和`other`)

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
- 沿着`_prev`一致往下找，直到没有前驱结点（来到的最初的变量）,这里通过递归调用实现

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