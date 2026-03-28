import math 

class Value:
    #定义核心节点
    def __init__(self,data,_children=(),_op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data='{self.data})"
    
    #前向传播与建图
    def __add__(self,other):
        out = Value(self.data + other.data,(self,other),'+')

        #定义和挂载闭包
        def _backward():
            self.grad  += out.grad*1.0
            other.grad  += out.grad*1.0
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data
        out._backward = _backward
        return out
    
    #非线性激活函数
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,))

        def _backward():
            self.grad += out.grad*(1-t**2)
        out._backward = _backward
        return out
    
    #构建拓扑图并实现反向传播
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in topo:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        #执行反向传播
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        