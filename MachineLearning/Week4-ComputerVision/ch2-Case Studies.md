# Case Studies



## Why look at case studies?

Recent years, researchers are focus how to put together these basic building blocks to form effective convolutional neural networks.

One of the best way to get some intuition yourself, is to see some of these examples.

So you need to read other researchers paper, after this class.

There are many famous and useful neural network frameworks, like

**Classic networks:**

- LeNet-5
- AlexNet
- VGG

**ResNet**

**Inception**





## Classic neural networks



### LeNet-5

We have learned this networks before.

People don't use padding in the LeNet-5 era, every time you apply convolutional layer, the height and weight of image will decrease.

The structure of LeNet-5 is as follow:

![image-20231019084119249](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231019084119249.png)

The parameters is just 60k, less than modern billion parameters.

This type of arrangement of layers is quite common.

The pure LeNet-5 paper has used a very complex method to calculate this convolution neural network, because the cost of computation, researchers use same channel for these convolutional layer. And they use Sigmoid/Tanh function but not Relu.

This is the hard one to understand about this paper, that after the convolution, it use a non-linear pooling method like sigmoid.

Just focus the section 2 about this paper, talks about this architecture, and just take a look at section 3.





### AlexNet

Similar to LeNet-5, but much bigger, has about 6000 million parameters, and use ReLu activation function, is better than LeNet-5.

The paper split the partial layers and training on the multiple GPUs.

And gives a new layer named Local Response Normalization, LRN, but not very useful.

![image-20231019113942754](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231019113942754.png)

### VGG-16

The neural has not too much parameters, just use a much simpler network where you focus on just having conv layers, really simplified these neural network architectures.

The filters is 3x3 and s is 1, all the same, so the Conv layer and Pool layer is very deep, the 16 is indicated there are 16 conv layers and full connected layer.

Although the VGG neural network is very deep, but the architecture is not complex, just the same shape filter, the weight and height goes down and the features goes up is very regularities.

![image-20231019115241172](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231019115241172.png)





## Residual Networks (ResNet)

The deeper neural network is hard to train and maybe exist the vanishing and exploding gradients types of problems.







































