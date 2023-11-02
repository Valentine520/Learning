# Basic



## 开始之前

两个重要的函数 在python中 dir(xx) 告诉我们工具箱中有什么 help(xx) 让我们知道这一切是如何使用的

python文件是为了传播方便和通用，通常适用于大型项目

IPython主要是进行调试

Jupyter Notebook 可以分块运行 更加自由 方便定位错误位置



## 数据加载

Dataset and Dataloader

Dataset 就像一个数据集合，指明你如何获取数据，获取哪些数据；Dataloader是在你获取到一些plain数据之后，用不同的方式压缩、汇总后得到一些新的形式，例如小批量随机梯度下降，会将数据随机打乱成小批量，直接使用即可

![image-20230926124156985](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20230926124156985.png)

通常，我们有训练数据集和测试数据集，所以我们需要将其分开，并且对于图像和标签，我们也需要将其组织成两个文件，一个是image，另外一个是label。

![image-20230926153109929](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20230926153109929.png)

同时，对于torch.utils.data 中的Dataset而言，你首先要继承这个类，是一个基类，你必须要重写 `__getitem__`，告诉Dataset class，该如何获取你数据集中的每个元素。你还需要重写 `__len__` 函数，能够返回数据集的数量，方便进行小批量随机梯度下降。

![image-20230926153423209](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20230926153423209.png)



所以我们需要获取图像的地址，才能使用`__getitem__`函数，所以我们可以利用python内置的os函数来获取某个目录下的文件名。

![image-20230926161946463](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926161946463.png)

同时你还可以使用os.path.join(xx) 用来拼接root和label的path

![image-20230926162152004](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926162152004.png)

Python 中有一些内置的函数 例如 `__init__(self, parameters)` 进行初始化操作 以及 `__getitem__(self, index)` 用来获取对应的元素，可以用一个例子来表示上述操作：

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926175638347.png" alt="image-20230926175638347" style="zoom:80%;" />

对于分别加载的数据集，还可以直接相加将数据集拼凑在一起

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926190829293.png" alt="image-20230926190829293" style="zoom:80%;" />

但是上面的操作仍然有些麻烦，我们可以使用更加简单的方式，就是将标签和文件分别存放，那样，可以将所有训练数据都保存在一个大文件中，然后将标签和文件分开存放就行。

![image-20230926191618236](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926191618236.png)

## TensorBoard

```python
"""Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
```

The official documentation is above, the tensorboard is write the log to event files.

![image-20230926214653207](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926214653207.png)



**writer.add_scalar**

The writer.add_scalar is also useful. The tag is the identity of these data, and scalar_value is the value you want to save, and global_step is the step value to record, so you need give these paramters, the horizontal axis is global_step and the vertical axis is scalar_value.

![image-20230926215159764](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926215159764.png)

![image-20230926215344736](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926215344736.png)

You need to use `tensorboard --logdir=logs --port=6007`, this --logdir need to be your tensorboard log file, you can run your program and finally you will get the logs you write to your tensorboard SymmaryWriter.

You can just fetch your scarlar value to this Writer after some steps, you can get the vivid image about your training process.

If you create SummaryWriter() not specify the file name, you can just use tensorbaord --logdir=runs to luanch the tensorboard.

![image-20230926220738963](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926220738963.png)

但是使用tensorboard 如果你想要重新绘制一些内容 但是你不该名称 会将你的图像重叠 相当于matlab中的 hold on

所以要注意tensorboard中存在的问题 可以为每一次新的tensorboard都创建一个新的内容

![image-20230927171325996](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230927171325996.png)



**writer.add_iamge**

The format is simaliar to add_scarlar, you need set your tag, and your global step (your training step), but the difference is the img tensor.

![image-20230927172033061](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230927172033061.png)

直接用PIL来读图像 但是读出来的内容不是tensor型 无法作为image

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230927172531380.png" alt="image-20230927172531380" style="zoom:80%;" />

所以可以使用numpy 或者 cv2 来将图像转换为数值，而不是一个jpeg

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230927173018058.png" alt="image-20230927173018058" style="zoom:80%;" />

但是如果直接使用 会导致报错 因为img格式的问题

![image-20230927173401248](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230927173401248.png)

```python
Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
```

C is channel, H is height, W is wide. So the CHW and HWC is different format of your tensor and ndarray. The default shape is (3, H, W), but we always use (H, W, C) to stand for a image, so just set the dataformats explicity, or you will get the error.

If you not sure your dataformat, just use image_array.shape to check, and chose appropriate dataformat.



## Transforms

Transforms主要是用来处理图像的 图像的变化等 `trochvision.transforms`

Transforms 就像一个大的工具箱 实际上是一个python文件 里面有各种各样的class 让我们可以使用合适的工具来解决问题

![image-20231001094141499](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231001094141499.png)

最常用的就是ToTensor 将图片和array转换为tensor 要求输入是一张图片 然后将该图片转换为tensor

![image-20231001094426484](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231001094426484.png)

注意 该ToTensor 是一个class类型 所以需要先创建一个对象 然后再使用该class的method

![image-20231001100439818](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231001100439818.png)

这里还有个小tips 如果你不知道一个函数需要哪些paramters 你可以使用ctrl + p

如上图，我们首先创建了一个ToTensor的对象实例 tensor_trans 然后使用该对象 使用内置的方法来将图片转换为tensor

无非是将图片经过某些变换 得到我们想要的图片变换

![image-20231001095601999](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231001095601999.png)

什么是tensor？ 和array有什么区别？

![image-20231001101042820](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231001101042820.png)

Tensor 包装了神经网络训练过程中所需要的一系列步骤 例如反向传播 将图像ToTensor 后再进行训练



anaconda 中的一些pkg是可以修改源代码的 因为开源 所以如果遇到一些无法解决的错误 可以 直接修改源代码

使用tansforms 则需要考虑清楚输入 输出 和作用 

![image-20231001110450438](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231001110450438.png)

很多transforms中的class都有 初始化 和 `__call__`

![image-20231001110958219](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231001110958219.png)

实际上 这里的 `__call__` 是一种重载 就是对原本的内置函数的重载 可以直接调用 无需使用class.hello(xx) 的形式 实际上差不多  一个是直接调用 一个是利用内置的function调用 

![image-20231001111203975](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231001111203975.png)

所以 对于compose 、 ToTensor等类来说 可以直接使用 **对象实例**进行调用 而不是直接调用

对于tensor的归一化处理 可以借助 transforms中内置的Normalization库来求解 让像素点集群的means为0 方差为1

但是实际上有何作用？**归一化可以让训练过程更快收敛 反归一化就是让图片更加可视化**

![image-20231001113305137](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231001113305137.png)

可以看出 归一化后图片尽管看起来不是那么友好 但是对与整个数据集而言 该图片有助于计算机进行训练 因为分布都很相似 可以用碗状的优化函数

![image-20231001113849424](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231001113849424.png)

对于某些图片 我们可能想要进行resize 但是注意resize的方式要恰当 要相同 对于tensor 和 PIL来说结果可能不同 PIL可能会被 antialias 抗锯齿化

你可以传入size 是一个sequence 那么就按照比例缩放 如果只是一个int 那么最小的为该size 其余边等比缩放

![image-20231007073535469](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007073535469.png)

具体的参数调用 则参照forward部分 输入可以是img或者tensor 返回值则同理 但是最好是用同类的内容来处理 否则img和tensor的最终效果可能会不同

![image-20231007074955628](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007074955628.png)

为了方便操作 我们最后还是要将PIL Image 转换为 Tensor的数据类型 用来模型的训练 以及传入给tensorboard

```python
# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
img_resize.show()
# img resize PIL -> totensor -> img_resize tensor
img_resize_tensor = img_tensor(img_resize)
print(img_resize_tensor)
```



上面这一堆操作实际上可以使用tensor.Compose来处理

Compose 就是结合多种对tensor的操作 然后将对应的操作应用到对应的tensor上

直接使用list作为参数传入即可

![image-20231007080139536](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007080139536.png)

对于Compose而言 一定要关注它的输入输出 保证输入和输出是匹配的 例如此处 trans_resize 需要PIL Image input 并且之后的totensor 也是需要进行一定变换的 

![image-20231007095057396](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007095057396.png)

transforms中还有许多对图像的变换操作 例如随机裁剪 RandomCrop 用随机的裁剪方法 裁剪出给定图像大小

**Transforms 的学习要注意关注输入和输出 以及多参考官方文档** 

使用type print debug等多种方式来查看当前变量结果的类型 获取返回值的类型 对于后续的编程极为重要

tensorboard对结果过程的可视化是比较重要的 并且非常好用



## 数据集 DataSet

pytorch 包含了多个模块 torchvision torchaudio torchxx 视觉 音频 等多个方面的处理

tensorboard and transforms all came from torchvision.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007104102569.png" alt="image-20231007104102569"  />

torchvision.datasets 中还包含了大量的数据集 对于COCO来说 通常会用于目标检测 例如对垃圾分类的识别

MNIST数据集 是教科书上的手写文字的数据集 

CIFAR-10 用来进行物体识别

torchvision.models 提供了大量的模型 已经训练好的神经网络 可以直接使用

torchvision的dataset如何与transforms配合使用

参数都比较通用 首先设置保存的root 然后设置训练集或者是测试集 然后指定对图像的变换 transform 然后是否下载数据 （帮助我们从网络上下载数据）

![image-20231007104530426](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007104530426.png)

简单打印一下获取到的单个数据 发现数据都是以img 和 tag的形式组成的tuple 这方便我们进行训练

![image-20231007105939930](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007105939930.png)

调试后发现 对于每个tag中的数字 实际上都有一个实际的类别与之对应 用一个index来代表label 用list来组成label序列

![image-20231007110109666](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007110109666.png)

在创建dataset的同时 也可以使用一系列的变换 组成一个transforms.Compose 然后作为transofrms的参数传入 在读取图像的同时对图像进行变换操作

![image-20231007112730846](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007112730846.png)

如果网络要求 你无法直接下载数据 可以使用迅雷等方式 下载好数据的压缩包 然后复制到对应的目录下 torchvision遇到该情况时会自动分析 解析下载的数据集

![image-20231007113607700](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007113607700.png)

下面是torchvision内部给定的一些数据集的下载地址和方式

![image-20231007113820331](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231007113820331.png)



## 数据加载 DataLoader

数据集就是上述torchvision类似的 一堆数据 然而数据是混乱的 对于训练来说是混乱的 所以我们需要让其有序 例如小批量随机梯度下降 我们也要让其自成一批量 一堆一堆

![image-20231008091417844](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008091417844.png)

看下官方文档 DataLoader 实际上有大量的参数 但是都有默认值 只有dataset没有默认值 我们需要传入的就是dataset

dataset可以是使用torchvision中的数据集传入 或者自定义一个dataset 前提是要必须有对应的函数定义 比如数据如何获取 数据在哪里等 这就是用自己的数据集制作 dataset

![image-20231008091739055](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008091739055.png)

对于num_works 来说 是指定并行程度 但是总是在windows环境下出现错误 

![image-20231008092423897](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008092423897.png)

注意 dataloader只是让数据更有序 例如打包 打乱 等 但是数据仍然是数据 如何从dataset中获取的仍然要从dataset中获取 但是loader中获取的数据是iterater的 然后获取的都是一个batch 而不是单个数据

![image-20231008093759407](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008093759407.png)

实际上就是打包成了一个高维的数组 

![image-20231008095703376](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008095703376.png)

在dataloader的sampler中也是随机抽样 表明对于dataset来说 是随机从中抽取内容

![image-20231008100004196](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008100004196.png)

利用tensorboard 可以可视化我们的数据集 利用writer.add_images 来add imgs

```python
writer = SummaryWriter("dataloader")
for index, data in enumerate(test_loader):
    imgs, targets = data
    writer.add_images("test_data", imgs, index)
```

对于dataloader 中的drop_last 实际上是为了消除最后一个batch的残缺部分 但是对于训练来说应该不会影响

如果drop_last 设置为True 那么就代表会消除最后一个部分的残缺值 也就是上下的两张图片的差别

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008102056682.png" alt="image-20231008102056682" style="zoom:80%;" />

参数shuffle控制的则是是否打乱数据集 如果设置为true 就是需要打乱对应的数据集 两次的数据是完全不同的 

如果设置为false 则代表两次的数据是完全相同的 不会被打乱的 打乱可以保证每次训练内容的不重复性 保证 数据集的随机性 否则模型训练得都是一摸一样的 不具有普遍性

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008102926727.png" alt="image-20231008102926727" style="zoom:80%;" />





## Nerual Network in Pytorch

The most nn model we use must be the subclass of the nn.Model, this is the basement of all neural network.

You can just define you own nn model, but you need init and rewrite forward function.

This is example define the foward function to do conv operation twice and do relu function twice, this is the forward propagation, deal  with the input and get the output.

The `__init__` function and foward function is very important, you must rewrite every time you want to make your own model.

You can define your Convolutioanl layer in your `__init__` function.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008104201752.png" alt="image-20231008104201752" style="zoom:80%;" />

If you want define your nn model simply, you can just do as follow:

Just make input plus 1 to get the output, if you want call the neural network, you must use tensor to be input, and you can get the output, also the tensor.

![image-20231008105257348](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008105257348.png)

When you create a nn object, just like `lover=Lover()`, will call the class init function to construct the object, `super().__init__()` to construct the basement of nn module.

Then you want feed your model with input value, this will call the foward function of this object, from input to get output.

If you want to optimized your nn model, this will call the backward propagation, use optimized function.

![image-20231008105450368](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008105450368.png)



Torch.nn.functional 实际上是一些小组件 细致入微的卷积操作 然而 Torch.nn 是更高层的api 由functional封装得到的

可以看到torch中的convolution操作包含了padding filters 以及最后的 bias 偏置，也就是一个完整的卷积层。

![image-20231020091653112](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020091653112.png)

要注意其中的参数 input的shape minibatch in_channels iH iW 不可缺少参数

而且你可以看到 padding参数还可以使用 string作为值 如果是valid 代表no-padding 如果是same 则可以自动让input和output维度一致

这种 *list 或者 *tuple操作都可以取出其中的元素作为参数传入

![image-20231020092609998](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020092609998.png)

多看官方文档 深度学习对代码要求并不高



`import torch.nn.functional as F` 也是常用 但是太过于原始 我们还是需要使用 `torch.nn` 中的Convolution layer 来搭建CNN

对于torch.nn 中的Convolution Layer 最常用的也是conv2d

![image-20231020095126573](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020095126573.png)

`in_channels` 代表的是输入的通道数 也就是上个图像的通道 `out_channels` 则代表的是多少个filters提取多少个features作为输出

你的kernel可以是square的，但是也可以指定不同的大小，当然对于padding和stride来说，也可以指定 tuple, 首先应用到 height dimensiion 之后用到 weight dimension

![image-20231020095521236](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020095521236.png)

卷积操作之后 图像就被加上了滤镜 但实际上对于计算机而言 简化了图片的信息 方便训练 以及方便区分图像的区别



![image-20231020101619827](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020101619827.png)

![image-20231020101626614](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020101626614.png)



大型网络是特别复杂的 在别人论文中的各种神经网络 同样也是非常复杂的 

你可以使用各种公式来推导 到底filter是多少 stride以及padding又是多少

![image-20231020102102282](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020102102282.png)



对于池化层而言 同样 nn 提供了各种类型的池化函数 在neural network创建的开始就定义这些layer

![image-20231020172659209](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020172659209.png)



对于Union来说 实际上是一种联合类型 Union(int, tuple) 代表你的参数可以是一个int整数 也可以是一个tuple 更加灵活 更加简单 这是深度学习函数中比较方便的部分

所以对于Pooling layer 而言 这里并没有那么多的参数 大多数参数都是固定的 

![image-20231020173145171](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020173145171.png)

similar to the convolutional layer Conv2d, the dilation control the exploded of the filter matrix, this is the official explanation.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020173614026.png" alt="image-20231020173614026" style="zoom:80%;" />

The ceil_mode control the steps of pooling, if True, the pooling operation can over the limitation of original image to compute the max value.

The default value is just false.

![image-20231020174006742](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020174006742.png)

这种报错原因是因为 torch.nn.MaxPool2 不支持long类型的数据 由于涉及到avg等操作 所以需要将tensor的dtype设置为float

![image-20231020174822568](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020174822568.png)

而且也不只是简单的float32 类型 而是 torch中的数据类型 要注意 这里的所有运算都是针对tensor的 所以任何的数据类型 数据操作 都是要在torch中进行调用的

![image-20231020175024900](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231020175024900.png)

非线性激活函数是非常简单的 不过设计这些函数却需要精心的思考 但是应用起来却十分简单粗暴

![image-20231022165355929](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022165355929.png)

ReLU的inplace 代表的是是否直接在原来输出进行变化 是否存在副作用

尽管ReLU函数的使用非常简单 使用条件也非常宽泛 但是仍然要注意 要将ReLU实例化之后才能继续使用

![image-20231022170025771](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022170025771.png)

无论你是自己定义一个nn 类别 还是实例化一个relu对象之后再操作 都需要使用对象进行relu操作 

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022170225728.png" alt="image-20231022170225728" style="zoom:80%;" />



各种各样的层 根据自己的需要来选择相应的layer 多种layer 组合你的特殊的神经网络

![image-20231022222616067](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022222616067.png)

nn 已经内置了大量的layers 随意组合 直到满意为止

![image-20231022223245083](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022223245083.png)

上述我们都了解了大多数常用的layer 还有最后一个就是linear

linear layer 就是全连接层 in 和 out 规定了输入输出的形状 其中有一个hidden layer 我们无需关心

![image-20231022223438655](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022223438655.png)

卷积操作之后 就可以直接展开 用全连接层进行操作 直接reshape就可以展开

![image-20231022223657536](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231022223657536.png)



对于fully connected layer 来说，展开最后得到是一个 一维的向量 前面的维度保留 关心的是最后一个尾部的维度 决定了hidden layer 如何来映射到 下一个layer

![image-20231023090932769](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023090932769.png)



如果不知道如何reshape 可以直接使用torch的内置函数 flatten 但是flatten会让最后的形状是一个一维的向量 这或许也是我们需要的

![image-20231023092012420](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023092012420.png)

对于含有最后一个不完整batch的内容来说，如果仍然进行相同的操作 会导致维度不符合，所以可以修改dataloader 或者修改dimension 让最终的dimension 的第一维度是关于batch的 带着batch进行一批量一批量训练

如何让网络自适应于input shape进行训练？

对于相应的操作 torch 已经将对应的model给我们训练好了 例如torchvision中的Models 所以我们可以直接使用这些已经训练好的模型 来处理我们对应的任务

![image-20231023092331467](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023092331467.png)



nn 中还有一个比较特殊并且好用的基础设施 Sequential 可以将你需要设置的各种layer 放置在这个Sequential中 组合成一个整个的模型 然后你的神经网络会分步骤 分批次地使用它 就相当于transforms中的一系列变换操作

![image-20231023092858330](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023092858330.png)



这就是cifar-10 的数据集对应的model 示例 对应的操作和维度都很清晰 我们完全可以复现

![image-20231023093044185](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023093044185.png)

上述的图解过程中 由1024 变为 64 这个过程中省略掉了一部分内容 我们可以自行添加layer linear1 



如果不知道参数的细节 实际上可以利用一些推敲和分析 来得到

![image-20231023105154846](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023105154846.png)



注意重新定义网络时的定义 

![image-20231023110315869](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023110315869.png)





如何检查你的网络是否正确？

可以随便使用一个测试案例 然后检查最后得到的shape是否满足你的要求

如果使用Sequential 你的model可能会更加简单 并且调用起来更加方便

![image-20231023111805442](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023111805442.png)

使用tensorboard 中的graph选项 可以看清你所训练的网络的整个结构 但是仍然不经常使用 不是很好用

![image-20231023112903610](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023112903610.png)



损失函数和反向传播 如何进行优化

Loss 就是用来衡量 预测和实际 两者的差别 让网络更加贴近于target 好的loss 可以精确定位问题所在 不会overfitting或者欠拟合 为我们更新整个参数提供依据 （反向传播）

![image-20231023113735197](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023113735197.png)

L1 Loss 以及 L2 Loss 等 有多种Loss fuction  

尽管Loss function的使用非常简单 但是仍然有一些参数值得我们注意

![image-20231023114649055](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023114649055.png)

Deprecated 代表已经弃用， 直接使用reduction就可以 最简单的就是到底是mean还是sum 或者 none 默认是mean 如果是None 那么就会导致输出的shape也是一致的 就是没有进行计算过的原始的比较结果

但是只有scalar才可以使用backward 反向传播 进行参数的更新

这种Loss 也常用 就是L2 norm的lossfunction 多了一个平方 更加合适

![image-20231023115147543](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023115147543.png)

对于分类问题 交叉熵是非常常用的 Cross-entropy

![image-20231023115421530](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023115421530.png)



如果要变得更小 那么对于正确类的预测概率应该比较高 也就是x[class] 以及对于整体的预测概率来说也应该比较低 避免出现大家都很高 啥都有可能的情况 （是否也可以通过softmax来避免？）

![image-20231023115700637](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023115700637.png)

这个交叉熵对输入输出也有一定的影响 

![image-20231023141446093](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023141446093.png)

根据需求来选择 loss function 然后注意其输入和输出

实际上的cross_entropy 输出结果会是一个(N, C) 的大数组 然后可以和targets一起计算出对应的损失函数

![image-20231023142200123](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023142200123.png)

然后你可以计算出loss 然后针对每一个batch 你可以将loss绘制成一个曲线 观察损失函数的变化 代表训练的效果

![image-20231023142518712](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023142518712.png)

损失函数在正向计算的过程中 也会计算各个树分支的梯度 然后方便进行反向传播 grad 用来降低loss 优化各个参数 得到最佳的模型

backward之后就开始应用一些优化器 优化参数 优化网络 达到更优的效果



官方文档中的优化器用法 就是先构造出优化器 之后再使用 构造时传入对应的参数 然后给出learning rate

![image-20231023143735200](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023143735200.png)

先清除梯度 之后再进行前向传播 然后再进行反向传播 之后再更新梯度 你的主角应该是你的网络 而不是你的loss 所以尽管optimizer 只是关注你曾经网络的parameters 但是它仍是在loss backward之后才能计算的

然后step之后更新梯度 这个是完全在loss 计算 并backward反向传播之后的

![image-20231023143843670](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023143843670.png)



任何一个优化器的背后都是比较复杂的数学公式 你需要设置的大多数都是参数 params 以及 learning rate 

对于背后的原理 可以去慢慢积累 先用后了解

![image-20231023144123395](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023144123395.png)



学习速率也是可以迭代的 最开始的时候学习速率可以高一点 越到后面 学习率应该越来越小 避免陷入局部最优解

你可以观察训练过程中的loss 但是实际上没有什么变化 因为对该数据学习的次数是比较少的 所以需要多训练几个epoch

![image-20231023150832844](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023150832844.png)

经过几轮的学习之后 loss 只会越来越小 设置多少epoch 这些都是需要斟酌的 要避免overfitting 也要避免欠拟合 
这些评估过程 可以使用一些table 来说明 看下variance 和 bias 到底是多少

所以深度学习的训练实际上是一个比较持久的过程

![image-20231023151113383](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023151113383.png)

![image-20231023151121838](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023151121838.png)

VGG-16 Model 也是非常常用的模型 但是已有的模型到底该如何修改 才能变成适合你自己所训练内容的模型？

VGG-16 是在ImageNet中训练得到的 所以利用该数据集来验证VGG-16 是十分明智的

此外该模型有一个预训练参数 weight

![image-20231023152631374](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231023152631374.png)



许多网络框架 都是存在局限性的 例如vgg16 如果我只有10个类别 最后的output shape就需要更改

所以大多数网络都是将vgg16 作为前置网络 然后在后面添加一些特殊的layer 让它适应我们的任务

使用 `add_module` 可以进行添加 

但是添加的位置有所考究 如果你是简单的直接在vgg16 的网络下面add 则会直接新建一层 但是如果想要在 classifier 或者 sequantial 这些已有的layers中进行添加 则需要细化到例如 `vgg16.classifier.add_module()`

![image-20231024010712135](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024010712135.png)

如果你想直接修改网络 你可以使用 `vgg16.classifier[0] = new_layer` 来完全创建一个新的layer 相当于各个layer就是对应dict下的一个sequential list 你可以直接用index来访问 然后修改

![image-20231024011126540](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024011126540.png)



模型创建好之后 如何保存和加载 对于已经训练好的模型 我们可以直接拿来使用 而不必每次都训练

保存和读取方法一：直接使用torch.save 和 torch.load 来保存和加载模型 后缀最好是.pth

![image-20231024091237201](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024091237201.png)

这样保存的就是整个模型 这是比较大的 如果你的模型较小 你完全可以使用这种方式 但是如果模型过大 难以保存 则可以使用 `torch.save(vgg16.state_dict)`，这个操作主要是用来保存模型的参数 但不是模型的全部 

可以节省空间 load之后 也只是得到一个参数列表 所以你需要先创建网络 `vgg16.load_state_dict(xxx)`，便可以进行上述已保存参数的加载

![](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231024093013009.png)

这里还存在一个小陷阱，如果你使用自己的model来save和load 会因为无法访问到你的自定义model类型而报错 这是正常的 因为分开文件之后 你的程序不知道你的网络是哪种结构 所以无法解析 

你只需要将你的网络放在load 的下面 你可以组成一个package 统一import

![image-20231024093002479](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024093002479.png)

如果你的网络是一个分类网络 例如有10个类别 但是你的target是1 这个时候你需要使用的就是crossentropy 而不是简单的 MSELoss 或者 L1Loss 这样是会引发维度不匹配问题的

这是训练的效果 Loss 的结果并不那么理想

训练3个epoch

![image-20231024103526485](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231024103526485.png)

训练1个epoch 

![image-20231024103629371](C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20231024103629371.png)

但是训练的次数确实可以提高模型的效果

tensor(10).item 才是实际的数字 所以如果你想获得纯数字 可以直接使用.item 方法

在training过程中也需要查看当前网络是否训练正常 我们可以在每个epoch中都哦添加一个test 判断训练效果 

甚至你还可以绘制出对应的图像 观察loss 曲线在test和train 中的情况

![image-20231024114034332](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024114034332.png)

这件事情tensorboard会帮我做 并且做的还比较好

![image-20231024114546199](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024114546199.png)



还有一个问题 我们的output是一个长度为10 的向量 每个元素都是预测为该类别的概率 所以我们该如何去找到output对应预测的到底是哪个类 可以使用argmax函数 原理非常简单 找到max 所对应的index就行了

然后计算出对应类别之后 该如何衡量整个过程的准确率？

preds == targets 就可以获得一个bool matrix 然后sum 就可以得到到底有哪些是对的 然后 / 总数即可 就是input targets的数量

将对应的正确率打印出来 这也是分类模型评估的一个重要指标 但是对于其他任务来说 直接看loss就行了 但是分类问题则需要参考正确率

![image-20231024154139753](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024154139753.png)



在训练之前 也许你可以将网络设置为训练模式 但是这没什么影响 除非是使用了Dropout技术或者BatchNorm

一般情况下只是一种形式 加与不加没有任何区别 eval模式仍然如此

![image-20231024154655585](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024154655585.png)

在test过程中关闭梯度 是比较明智的 使用 with key words

![image-20231024155308202](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024155308202.png)



cpu的训练太慢了 还是要使用gpu进行训练才是正解 但是只有模型 数据 和 损失函数存在cuda 也就是可以调用gpu 

数据就是dataloader中的imgs 以及 targets 模型就是创建好的模型的对象 Lover().cuda()

![image-20231024165720711](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024165720711.png)

将 imgs 和 targets 都转换为cuda 模式

![image-20231024165927750](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024165927750.png)

但是最好的是判断是否可用cuda  使用 `torch.cuda.isavailable()` 进行判断 如果不可用则使用cpu模式

此时我们的cuda 是不可用的

![image-20231024170224356](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024170224356.png)



**在训练的过程中如果全都要打开GPU 那么训练集和测试集都要使用GPU加速 否则会报错**

![image-20231024171411803](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024171411803.png)

上述的操作需要对数据和网络同时启用cuda 但是有时候我们想切换设备 例如哪一个cuda 无法精细化 并且要挨个处理 所以我们使用.to(device) 进行处理即可

device = torch.device("cuda") 这样你想切换任何的训练设备都可以 提高代码的重用性也是重要的编码理念

![image-20231024175723178](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024175723178.png)

如果想要在没有cuda 的条件下保证模型的正常训练 可以添加一个if 判断语句 这是很常用的写法

![image-20231024183505014](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024183505014.png)

学完这些就入门了 但是还有很长的路要走 

能看懂代码 能修改代码 从论文到代码 一条漫长的深度学习旅程

看别人的参数列表代码 这里的required = True 要求这个参数必须传入 但是通常我们将其改为default更加rubost

![image-20231024194251481](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231024194251481.png)























