# Bigrams 统计建模

## 读取原始数据集
```python
words = open('names.txt', 'r').read().splitlines()
```
- `open()`打开文件，第一个参数是文件路径，第二个参数`r`表示`read`标识模式，表示只读不修改
- `read()`是挂载在文件对象上的一个方法，将文件内的所有内容全部读进内存，变成一个连续的字符串，用`\n`连接在一起`"emma\nolivia\nava\n"`
- `splitlines()`按行切分，寻找换行符进行切分并去掉换行符，这步会构成一个列表`['emma', 'olivia', 'ava']`

## 获取二元组并计数
```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0) + 1
```
- 先构建字典`b`未来键为元组，值就为这个组合出现的次数
- 遍历每一个单词
- 人为的设置开头和结尾，并将整个单词拆为列表后拼接`['<S>', 'e', 'm', 'm', 'a', '<E>']`
- `zip(chs,chs[1:])`滑动窗口提取相邻二元组`Bigram`,`zip()`中两个可迭代对象（list）逐个元素对齐.（注意`zip()`语法糖会自动对齐短的可迭代对象长度，也就是`chs`中多出的那个会由于没有配对的元素而直接丢掉）
- 逐个提取出每一对，安全的计算到字典中，这里`.get(key,default)`会先判断字典中是否有这个元组键，如果没有则返回0，防止没有的时候返回`KeyError`


## 字典排序
```python
sorted(b.item(),key = lambda kv: -kv[1])
```
- `b.item()`将字典转换为一个包含元组的列表`[(('<S>', 'e'), 1531), (('e', 'm'), 582), ...]`
- `lambda kv`: 定义了一个匿名函数，kv 代表传入的每一个元组（比如 `(('<S>', 'e'), 1531)`）
- `kv[1]` 提取出元组里的第 1 项（注意索引从 0 开始），也就是`频次 1531`,加个负号 `-`：`sorted` 默认是从小到大排（升序）。给频次加上负号后，频次越大的数（比如 `-1531`）反而变得越小，就会被排在前面，这就极其巧妙地实现了降序排列

```txt
[(('n', '<E>'), 6763),
 (('a', '<E>'), 6640),
 (('a', 'n'), 5438),
 (('<S>', 'a'), 4410),
 (('e', '<E>'), 3983),
 (('a', 'r'), 3264),
 ······
]
```
发现排在最前面的可能都是 `('a', '<E>')`这种，说明以字母 `'a' `结尾的名字在数据集中极其庞大！



## 字典到矩阵
字典虽好，但无法做数学运算，在框架本质里都是数字运算，这里需要做一些处理，转化为为矩阵，便于后续的运算
```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0 #将两个特殊字符<S><E>合并为'.'

itos = {i:s for s,i in stoi.item()}
```
- `set(''.join(words))`：`''.join(words)` 把所有几万个名字粗暴地连成一个超长的字符串,`set()`将字符串去重（剩下的即是26个字母）
- `stoi(String TO Integer)`:构建字符到矩阵索引的映射`enumerate` 会吐出 `(0, 'a'), (1, 'b')...`。我们让**字母做`Key`，把索引` +1` 做 `Value`**（即 `'a': 1, 'b': 2`）,这里的`+1`是为了把0号位置留给特殊字符`.`
- `itos`反过来，数字做Key，字母做Value，方便后续模型算出数字结果之后翻译回人类能看懂的字母

### 矩阵构建
```python
import torch
N = torch.zeros((27,27),dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1] #获取矩阵索引
        ix2 = stoi[ch2] 

        N[ix1,ix2] += 1 #频次添加
```
在完成了字符到索引数字的映射之后，就可以将各个`bigram`的频次关系构建到矩阵中


## 概率转换
```python
P = (N+1).float()#+1是为了平滑处理，放置log(0)对数值的影响
P /= P.sum(i,keepdims=True)
g.torch.Generator().manual_seed(214783647)  #设置随机数种子，保证抽出的5个都是一样的，便于调试对比
```
特别注意，第二行使用了广播机制，通过`.sum(1,keepdim=True)`将矩阵跨列相加（即每一行相加），再将`(27,1)`广播成`(27,27)`进行相除，就得到概率矩阵

## 生成预测
```python
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos(ix))
        if x==0:
            break
    print(''.join(out))
```
- 外层循环`range(5)`抽取生成五个名字
- `ix=0`作为关键起点，每次生成名字要从其实符开始
- `p = P[ix]`： 拿着当前的字符索引（比如现在是 0），去概率`矩阵 P `里把`第 0 行`整行提出来。这一行就是“所有字母作为名字首字母的概率分布”。
- `torch.multinomial` **核心抽样器**:这是 PyTorch 的内置函数。它就像一个不公平的转盘，转盘上 27 个区域的面积大小由 `p` 里的概率决定（比如 `a` 的区域很大，`x` 的区域很小）。函数转动转盘，抽出下一个字符的数字索引。`.item()` 负责把抽取结果从张量剥离成普通的整数。
- `itos[ix]` 查表翻译:拿着刚抽出来的数字，去“数字到字母”的密码本里查出对应的英文字母，塞进 `out` 列表里。
- `if ix == 0: break` 触发刹车： 如果命运的齿轮转动，恰好又抽到了 0（也就是结束符 .），说明模型认为这个名字该结束了，直接打破死循环。最后，用 `join` 把列表里的字母拼成一个完整的字符串并打印出来。

```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
```
结果杂乱无章，故而仅依靠统计是无法给出很好的名字，接下来将采取神经网络的方法进行