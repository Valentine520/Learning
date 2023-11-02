# Convolutional Neural Networks

**Computer vision problems**:

![image-20231017091727307](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017091727307.png)

And the input maybe very huge, the image is 3 dimension, the weight*height * (rgb). If you just have 64 * 64 image, the X, the input features single dimension is 12288 length. If the image is 1000*1000? the dimension of input feature will reach 3 million.

This is just one image, but you have lots of these to train, and your neural network maybe very large and complex.

**The parameter of the complex NN model is very huge, almost 3 billion, very hard to train this model, so you need to do convolution operation**, which is one of the fundamental building blocks of convolutional neural networks.

![image-20231017092111403](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017092111403.png)



## Edge detection example

In the image, you want to detect the edge, the vertical edge, the horizontal edge, how you can do this?

![image-20231017092458608](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017092458608.png)

Consider another simple vertical edge detection.

If you have 6x6 image, you want to do convolution, transfer the image into a new matrix, you can use the convolution kernel, this is also called filter.

You can see the convolution operation is just use the filter to calculate a new value, move the kernel left and right or up and down, you can get the whole matrix of 4x4 dimension, this is convolution operation.

![image-20231017094614205](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017094614205.png)

If you implement the convolution operation in program, you can use the built in function, such as the python's conv_forward, and the tensorflow's tf.nn.conv2d and the keras Coonv2D.

Why this convolution operation can detect the vertical edge?

The vertical edge, we assume the left is bright and the right is dark, so if we use the filter which is left bright and middle 0 and right dark, when you do convolution calculation, the left will be dark and the right will also be dark because the value will final get 0. But the middle, will be bright, because the left value maybe positive, and the 0 and -1 * 0 is also 0, so the final value is just positive, so this is bright, this is the vertical edge.

You can see the vertical edge in the following situation, if your image is 1000x1000, the result will more clear.

![image-20231017095043878](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017095043878.png)

This is the simple convolution operation, different filter will get different feature, and get different result.



## More edge detection

Like the vertical kernel, the horizontal kernel is also the same.

Because our image just 6 x 6, so the final edge will be a little big, this is not problem.

![image-20231017100707293](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017100707293.png)



But the filter above is not the best, the researcher find if change the weight of the middle the filter will work well, just like the sobel filter and the schorr filter.

The fixed number is not always the best solution for each image, you can use the convolution filter just as the parameters which you can update in your backward propagation. Let them detect the $45^\circ$ and other angle edge not always the vertical. This is the CNN(Convolution Neural Network)

![image-20231017101154558](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017101154558.png)



Before we learning the CNN, we must know the Convolution's important concept, like padding and other development of Convolution.



## Padding

The dimension of image which be convoluted is the (n-f+1, n-f+1), the n is the original image size, and the f is the filter dimension.

![image-20231017102242364](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017102242364.png)

There are **two downsides** of the convolution operation.

**First**, Whatever your filter is, your image will be shrink after convolution, your image may be shrinks down to 1x1, it's very bad.

**Second**, because of the computation characteristic, the pixel in the corner will be ignore, and the middle pixel will be used very frequency. It means you're throwing away a lot of the information near the edge of the image.

![image-20231017102554461](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017102554461.png)

In order to solve the downsides above, you can **just padding the image,** make your corner pixel no longer the corner, and expand the dimension make sure the convolution operation cannot change the image dimension.

You can just pad with 0, the p is the padding range, if your p value is 1, it means you will increase 2 dimensions of your image, left and right will also increase 1.

The final image dimension becomes to (n+2p-f+1, n+2p-f+1).

Now the original corner pixel can be used overlapping, this downside will be reduced.

![image-20231017103153452](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017103153452.png)



## Valid and Same convolutions

**valid just means no padding**, the dimension will (be n-f+1 x n-f+1), the 6x6 and 3x3 filter will get 4x4 image.

**same is pad so that output size is the same as the input size.** $n+2p-f+1 = n$, so $p=(f-1)/2$, you can calculate the padding size p, to get the same dimension output.

**f is usually odd**, let you can calculate p to do same convolutions and the filter **has only single central pixel**.





## Strided convolutions

padding and strided convolutions is the two basic blocks of convolution neural network.

above we just use stride=1 to do convolution, but the stride also can change.

Now the whole output dimension can calculate as **floor($\frac {n+2p-f} {s}  + 1$)**. 

**The floor function is to prevent the dimension is not an integer, your filter must whole in the original image.**

![image-20231017105315460](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017105315460.png)



Summary of convolutions output dimension:

![image-20231017105346834](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017105346834.png)

In the convolution in match textbook, there must be a mirror flipped, and what we do in previous convolution is just called cross-correlation, it doesn't matter, this will not affect the training result in our CNN.

![image-20231017105934589](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017105934589.png)

In the signal processing, the convolution and cross-correlation actually different, but just a mirror flipped filter, in our CNN, the convolution filter can be learned in the backward propagation, so whether the convolution or the cross-correlation is doesn't matter.

So machine learning don't need these mirror flipping, just calculate just train.





## Convolutions over volumes

The real image is 6x6x3, the final 3 is the 3 color channels, so different with above 2 dimension filter, the convolution on RGB images need 3 dimension filter.

the height and width and channels, the channels must be the same, and the final output dimension is just 2 dimension, just 4x4, not 4x4x3(, how to keep the same dimension?)

![image-20231017111428211](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017111428211.png)

How to calculate the convolution output?

Let the 3 overlapping filter combine into a cube, now you must care about your red 、green 、blue 3 channels filter, you may want to detect the vertical edge of red, so you can set other channels all 0.

You can use the any size filter but you must keep the channels all the same.

![image-20231017112626112](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017112626112.png)



### Multiple filters

If your want do many other detection during single detecting, like vertical detection and horizontal detection, this is called multiple filters.

You can use two different filters to do convolution simultaneously, and get the different result, and stack them into a cube, you can get the 4x4x2 cube as result.

![image-20231017113139908](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017113139908.png)

The output size can be calculated as the previous non-RGB images, the padding the stride, the p, but just keep the n_c is the same. The channel 3 is also be called the depth of this 3D volume in some related papers.

![image-20231017113513576](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017113513576.png)



## One layer of convolutional network

As the normal neural network, the forward propagation, just input is a image, a 6x6x3 matrix, and the w[1] is the convolution operation. 

The 6x6x3 input in, the 4x4x2 output out.

This is single layer of CNN. 

If we have 10 features to extract, the final output dimension maybe 4x4x10.

![image-20231017152552434](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017152552434.png)

Calculate the number of parameters of this CNN.

The number of parameters is depend on your filter dimension and the features, whatever your original image is, the dimension is also fixed.

![image-20231017153000480](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017153000480.png)



### Summary of notation

The Weights is f[l]xf[l]xnc[l-1]xnc[l], maybe you think this is a little hard to understand, the nc[l] is the filters in layer l, the nc[l-1] is just the previous image's channel.

The activations just the a[l], and the n[l] number can be calculated by the previous layer n[l-1].

The channel is a little complex to understand.

![image-20231017154112211](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017154112211.png)


## A simple convolution network example

We learned the single layer convolution network in the above, now we need to stack these into a deep convolution network.

During the convolution operation, the dimensions of the image decrease very fast, finally, we get the 7x7x40, just 1960 cell, and you can feed this into your softmax regression to get the final result.

The final step is our normal neural network doing, but the previous layer is the convolution operation.

With the help of convolution operation, your input complex decrease very fast, so you get the final result quickly.

![image-20231017155212713](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017155212713.png)

**But a lot of the work in designing a convolutional neural net is selecting hyperparameters** like these, deciding what's the filter size, what's the stride, what's the padding, and how many filters you use.

The image size change from 39 to 37 and to 17 then 7, whereas the channel number increasing, from 3 to 10 and 20 and 40, but the total size become smaller.\

### Types of layer in a convolutional network

- ![image-20231017160240741](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017160240741.png)

Although it's possible to design a pretty good neural network using just convolutional layers, but almost every research will add the pooling layer and fully connected layer.

The pooling layer and fully connected layer is simpler to design for the convolutional layers.



## Pooling layers



### Max pooling

Just split the whole region into some small region, and use the max number to stand for this region.

the max operation does is so long as the feature is detected anywhere in one of these quadrants, it then remains preserved in the output of Max pooling.

If this feature is detected anywhere in this filter, then keep a high number, if not, the max number may be small, the feature doesn't exist, this is intuition to max pooling.

The pooling is very useful, although it may a little hard to understand why it goes well.

![image-20231017162040438](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017162040438.png)

The pooling operation have 2 hyperparameters but don't need to learn,  this is a fixed computation which gradient descent will not change.

The output dimension can be calculated by previous formula of convolution,  **floor($\frac {n+2p-f} {s}  + 1$)**.

The padding operation is not useful in the pooling operation. 

If you have multiple channels, you can just repeat the operation of one channel for every channel.

![image-20231017162925276](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017162925276.png)

Max pooling is used frequently, but there still some other pooling strategy, like average pooling.



### Average pooling

Not always use, just using when your neural network is very deep, you might use average pooling to collapse your representation from 7x7x100000, get 1x1x100000.

![image-20231017163324085](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017163324085.png)



### Summary of pooling

The f=2, s=2 is often used, this has the effect of shrinking the height and width of the representation by a factor of two.

The padding is not used in max pooling. And no parameters to learn in backward propagation. Max pooling is just a fixed function that the neural network computes in one of the layers.

![image-20231017163353138](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231017163353138.png)

## Neural network example

Some researcher often treat the conv and pooling layer as a whole layer.

The fully connected layer is just the layer, which previous are connected them closely, just **like the normal neural network**.

Let's see the LeNet-5, there are two layer's of convolution and pooling, and then flatten the 400 cells, then step into the fully connected layer, the FC3 and FC4, finally, use the softmax regression to get the final result.

This NN model have lots of hyperparameters, and one common guideline is to actually not try to invent your own settings of hyperparameters, just choose an architecture that has worked well for someone else.

As you go deeper, usually the height and width will decrease, whereas channels will increase.

![image-20231018101751475](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018101751475.png)

During the going deeper, the activation shape and parameters shape is also very complex. 

Actually, after the conv layer maybe a relu layer, then the pool layer, this is a whole layer, convolution layer.

![image-20231018112450186](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018112450186.png)

The whole parameters and activation shape change is as the following table:

For the parameters, the pooling layer just have fix f and s, but don't need to update, so the number of parameters is 0. And the conv layer has just a little parameters, the most parameters are in the FC layer.

For the Activation shape, the features or channels increase but the weight and height decrease, this can reduce the activation size, the activation size decrease from 3072 to 400, so you can use the normal neural network to get the final result.

In my opinion, **the convolution operation is just** **extract the useful information, or the features** from a huge image, reduce the complex of the input, let our FC layer can be trained by just a mall dimension input, but huge data set.

![image-20231018102852651](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018102852651.png)

Some researchers are want to put these building blocks together to build very effective neural network, the only effective way is just reading the successful case other people do.





## Why convolutions?

Consider the convolution layer, if you replace the convolution operation and just use fully connected, your parameters will reach 14millions, but if you use the convolution operation, your parameters is just 156.

There isn't need so much large parameters, you just need extract the features from image, so we use convolution to simplify these parameters is reasonable.

![image-20231018105350301](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018105350301.png)

### Why the convolutions have just few parameters?

**First,** One of the most important reason is the **Parameter sharing**.

If you use a vertical edge detector as your filter, **you can apply to every where of your image just use one filter, the parameter is sharing, is same**. You can apply the filter to left corner, the middle, almost every where, you needn't create any new parameters just like the FC.

And this same parameters is useful to extract the features.

![image-20231018105737928](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018105737928.png)

**Second,** **the Sparsity of connections**.

In each layer, each output value depends only on a small number of inputs.

The output left corner value is just related to the input left corner 9 features, all of the other pixel value will not effect the specific output, doesn't like the FC layer, the output is related to all of the input features.

![image-20231018110345863](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018110345863.png)

The convolution neural network can use these two mechanisms, to reduce the parameters, which allows it to be trained with smaller training sets, and prevent over-fitting.

putting them together, you can see the convolutional neural network whole construction steps.

![image-20231018111020143](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018111020143.png)





# Exercise

For the convolution operation, the padding is very important, before we use the p=1, add 1 dimension for every axis, but you can specify the axis and the padding width, just like the `np.pad`.

The pad_width indicates the padding range, you can use `(before1, after_1)` to specify the different padding length of left and right. And for each axis, you can also specify the padding width, `((before1, after1), (before2, before2))`. It's hard to understand, just calculate use your pencil.

![image-20231018114120321](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018114120321.png)



![image-20231018160830152](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231018160830152.png)

Attention to the dimensions of these activation matrix.



















