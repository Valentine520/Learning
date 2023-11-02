# 基础内容

#### isinstance(value,(type1, type2))

在程序运行中可以使用这种方式来检查变量的类型

#### 

#### python中一些皆对象 

有些基本类型 a = "hello" 实际上 a.upper() a.lower()  等 都是可用的 

a.count('a') 可以用来统计str中 'a' 字符出现了多少次



#### python对类型管理比较松

有时候并不关心一个类型的真正类型 只要可以满足特殊方法就行

例如 可迭代 

##### isiterable(obj) 用来判断该对象是否可以迭代 

或者isinstance(obj, list) 也可以判断是否是列表 

因为只有可迭代的内容才能被放在range(xxx) 中

if isinstance(obj, list) or isiterable(obj) :

​	obj = list(obj)

如果不可迭代 要转换为可迭代类型



#### is not / is 也可以进行类型判断 以及对None的判断



#### 如果不想使用反斜杠来进行escape 可以直接r'hello\ world' 代表raw 



#### 字符串的格式化可以先保存一个模板对象 

template = '{0:.2f} {1:s} are worth US ${2:d}' 之后template.format(4.5560, "Argentine Pesos", 1)

就可以利用已经保存好的一种模板 传入值之后来格式化一个对象 

0: 代表处理第0 个参数 也就是第一个 其后的其余参数均以此类推



#### byte 也是一个标准类型 可以decode('utf-8') 也可以encode('utf-8')

之后就会得到一个 b'hello world' 这就是代表以byte的方式表示一个str

因为历史遗留问题 许多内容都会遇到格式混乱问题 曾经的unicode 到 如今的utf-8 编码形式发生了较大的变化



#### datetime 处理时间的自带模块

datetime(2022, 10, 8 , 10, 23, 34) 在创建的时候就键入所有的时间信息 

dt.day dt.date() 等 都可以通过内置的属性来访问

dt.strftime("%m%d%Y%H:%M") 月日年 小时 分钟

对于格式来说 因为Month 和 minute 是相近的 所以对于这两者而言 m是月 M代表minute

dt.strptime(str, "%m%d%Y") 根据前面str的格式来选择后面format的组合 

就是str转换为datetime类型 方便计算和管理



#### 值得一提的是 python的对象方法 实际上不会改变本身的内容 

dt.strftime(xxx) 也是以return的方式返回一个新的内容 原来的内容仍然不变



#### range 作为python的主流for 对象 range(start, end, step) 

同时，一切可以迭代的类型都可以使用for循环 实际上 range生成的内容也是一个迭代器

这就让a = range(xxx) 成为可能 之后 for i in a: xxxx 

引入迭代器这个概念 可以让许多内容都可以被遍历



#### 三元表达式 true_expr if condition else false_expr

如果为真 则执行true 这个表达式 如果为假 执行后者

这种方法可以在一行之内就可以得到赋值 但是实际上可读性不佳

lambda表达式似乎也是如此



#### lambda函数

lambda x, y: x*y 实际上 就是定义了一个匿名函数 

这样做可以使用一个变量来接受赋值 当然也可以直接传入参数中参加计算 例如map



#### 高级用法就是将lambda表达传入函数中 作为参数

##### map(function, iterable, ...)

返回一个迭代器 可以自己range 或者直接list(res) 转换为list 类型 

实际上 前面的function指定的是对后面的内容 如何操作 

iterable如果有两个 那么lambda x, y :xxx 这就是合理的 定义的是对其中每个元素如何操作

是map映射的原理 如果只是一个 lambda x: xxxx 这才是合理的



##### reduce(function, iterable[, initializer])

*（仅在python2中存在）*

**参数：**
function  ----> 函数，有两个参数
iterable  ----> 可迭代对象
initializer ----> 可选，初始参数
**返回值：**
返回函数计算结果。

通俗来讲 这个函数就是进行累积计算 但是不是简单的累加和

例如function 指定为 lambda x,y : x + y 那么就是简单的相加运算

如果为 lambda x, y : x * 10 + y 那么就是将list还原为曾经的大数字





##### sorted(iterable[ ,key[, reverse]])

参数说明：
iterable  ----> 可迭代对象。
key        ----> 主要是用来进行比较的元素，如果只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
reverse  ----> 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
返回值：
返回重新排序的列表。



目前对于cmp的比较以及对应的比较函数都已经被废弃或者删除

只需要指定key就可以进行比较



#### 

#### 如果有多个返回值 可以使用*rest 来接受

values = 1,2,3, 4,5

a, b, *rest = values 这样 尽管之后的内容没有接受完 也会由rest一并接收

a, b, *_ = vlaues 将后续值都丢弃



#### 对于列表的操作

extend可以添加多个 append 只是末尾添加一个

如果想要在for value in xxlist 的时候同时知道索引 可以使用enumerate(list) 函数

如果要同时遍历多个序列 可以使用zip函数

for i, (a, b) in enumerate(zip(seq1, seq2)):

两者相互配合 可以完成更加强大的任务

for range这类型的操作 在python中是非常常见的 因为是为了让工作自动化进行



zip的压缩 可以使用zip(*zipped) 来解压缩 分解为多个list 而不是以元组形式存在的

zip的结果实际上是一个地址 所以需要使用for 来遍历 

如果要分别取出值 则需要使用tuple解包的形式 

for (v1, v2) in zip(list1, list2):



#### 对于字典 

empty_dict = {}

d1 = {'a' : 'some_value', 'b' : [1, 2, 3, 4]}

**del** 用来删除键值对 dict.keys() / values() 给出迭代器 但是没有特定的顺序 是乱序的

**update** 用来合并两个字典

**dict.get(key, default_value)** 可以设置对于字典查询或者访问的一个缺省值

同时 内置的**dict.setdefault(key, [])** 可以在无内容时直接设置

通过对缺省值的操作 可以减少许多对于初始化情况的考虑





#### 好用且神奇的推导式

[expr for val in collection if condition] 

这里expr 实际上是拿去的后面val的内容 expr(val) 对遍历出的val进行操作 且存在条件限制

当然 dict 也可以

{key: val for xx in dict if xx }



如果推导式返回的expr是一个list 那么还可以进行for 的嵌套

![image-20230808073927408](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230808073927408.png)

实际上 这个地方的嵌套里面 是顺序的 也就是说无论前方是何内容 例如name 

只要后面存在就行 不管前面的位置在哪里 

name for names in all_data 这就是其中一个 但是没有完 我们还要继续深入 for name in names if xxx



#### 函数中的global

如果想要内部改变外部内容 将其声明为global就行  global a 代表是全局的



#### 函数也是对象

那么就可以像简单的数据一样 直接将函数传入 或者直接使用某个函数来操作就行 没有很复杂的限制

相当于就是使用函数指针来操作一个函数

clean_ops = [str.strip, remove_punctuation, str.title] 

在之前的map中 我们已经领略过了



#### 生成器和迭代器

迭代器就是一个可遍历对象 for i in range(xxx):

生成器就是一种惰性的可遍历对象 只有使用时才会生成

例如函数的返回值中如果有多个 可以使用list 也可以使用yield来制造一个生成器

![image-20230808081453884](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230808081453884.png)



和列表表达式类似 生成器表达式可以作为函数参数直接传

gen = (x ** 2 for x in range(100)) 实际上只是[] 变成了()



#### error catch

try :
		xxx
except ValueError:
		xxx
finally: 
        xxx

这就是整套异常捕获和解决的逻辑 有时候你失败了 但是总是希望关掉某些内容 避免不必要的混乱



#### 文件操作

python的文件操作是比较简单的 

f = open(file, "w") 可以指定打开文件的方式 如果不存在则会创建 (w)

with open(path) as file:

f.close()

f.read(1) 

f.seek(xx) 

seek是用来定位光标 read则是用来一个一个读

readlIne、writeline、write、flush、tell 告知当前光标的位置 方便seek





# Numpy

numpy实际上是一个用c写的 用python来调用的一个库 针对的是快速向量计算 自身支持全量计算





## numpy基础类型

numpy中最基础的是ndarray 就是一个多维度的数组 矩阵、向量均是ndarray 
有**shape**  **dtype** **ndim** 

numpy的操作大多都是可以传播的 对一个ndarray操作 实际上是对整个进行操作

shape和dtype等都是比较关键的 就像三维 在进行操作时要注意shape和type的一致性

zeros emtpy ones eye identity 都可以快速创建一个新的ndarray 

zeors_like empty_like 等 可以根据形状来生成一个新的对应类型的ndarray

astype 方法可以显示进行类型转换 在创建数组的时候也可以显示声明dtype



**ndarray的切片索引是一个view 也就是说 不会新创建内容 而是会引用 **

如果需要完全新创建一个ndarray 那么需要使用.copy() 显示声明复制



维度增加时 有些东西就不方便理解了

![image-20230808102534959](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230808102534959.png)



对多维度的操作会进行扩散 arr[:2, :1] = 0 不会造成维度不匹配 而是会进行扩散 这就是python

同时 还可以使用一个维度相同的bool 值对array进行布尔索引

![image-20230808114401723](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230808114401723.png)

这样，就可以利用true 以及 false 进行索引 处理布尔值

bool索引之后还可以利用 ， 分隔开进行切片索引

and 和 or 对于numpy并不适用 而需要使用 传统的C语言中的bool运算 这和numpy的C基因有关



对于神奇索引来说 arr[[1, 2, 4]] 索引的行 arr[[1,2 ,4], [1, 2]] 索引的则是交叉的元素

arr[[1,2]] [[1,2]] 首先索引行 之后索引列 

arr.T 可以轻松完成矩阵的转置 或者 np.dot(ndarry, ndarry) 就可以实现矩阵的乘积



arr.swapeaxes(1,2) 可以完成轴的交换 行变换为列

arr.transpose((1, 0, 2)) 同样可以用来完成轴的置换

这个地方实际上有点拗口 目前暂时没有想到用处



numpy中的许多操作 都是逐元素进行的 np.sqrt(ndarray) ...



对于numpy中的带有条件逻辑的数组操作 也可以使用where 来加速

![image-20230808173535963](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230808173535963.png)



np.where 后面还可以跟一个标量 cond为true 则为a false则为b 

实际上 无论是标量还是数组 满足的值都应该是一个根据condition来索引的value 

![image-20230808173753445](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230808173753445.png)

这种用法是比较常见的 如果为负数 则如何 如果为正 又如何 可以快速批量处理数据



还有一些比较好用的统计方法 例如 mean sum min max cumsum 等等

np.unique(xxx) 可以用来保留单一内容 处理重复出现的值

np.in1d(x, y ) 可以计算x的值是否包含在y中 得到一个bool array



numpy.random 可能是用得比较多的内容了 

randn rand randint normal beta 等都是生成一些随机数 

seed 可以自己设定 



numpy 还提供了线性代数常用的一些method



至此 python的复习 以及numpy的复习 就告一段落 

![image-20230808174745818](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230808174745818.png)



许多东西都是记不住的 所以要常常翻阅 用到的时候就知道该如何用了 别担心 别焦虑

