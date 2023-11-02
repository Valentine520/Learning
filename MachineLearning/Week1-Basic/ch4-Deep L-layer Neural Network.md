# Deep L-layer Neural network



## Notations and basic concept

What is a deep neural network? The more layer, the network is more deep.

If you don't know how much layer your model need, you can just use logistic regression and then with 1 hidden layer and 2 hidden layers and more and more layer, just make your network can describe your problem well. This is a hyper parameter which you can try variety of values.

![image-20230910222022753](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910222022753.png)

The notation in deep learning is also important.

We just use the capital L equal to 4 to indicate there are 4 layers in this neural network, and the n[l] just like n[1] = 5, indicate the unit number in the special layer.

a[l] indicate the activation functions in layer l, w[l] also the weight in the special layer.

Especially, the first layer activation function x is equal to a[0] and the output value y_hat is equal to a[L], because they just equal to themselves.

![image-20230910222712891](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910222712891.png)



## Forward and backward propagation



### Forward propagation

Just as the logistic regression calculate method, the neural network chain from input X to output Y_hat.

![image-20230910224827995](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910224827995.png)





### Backward propagation

![image-20230910225252521](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910225252521.png)



### Forward propagation and backward propagation.

When you compute the Loss function from input to output, this is forward propagation, and during this forward propagation, the z[1] z[2] z[3] can be calculate, and calculate the dz[3] dw[3] db[3] and so on.

So the compute graph realize the forward propagation and then will must compute the derivative for backward propagation during it's forward propagation.

And you can just implement vectorizing to make the da[1] stack into a matrix dA[1].

![image-20230910225825338](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910225825338.png)



These all kinds of formulas is just computing the derivatives especially in backward propagation, don't worry, maybe it's very serious for you, just practice, many duplicated work we don't need to implement every time.

Lots of complexity of your learning algorithm comes form the data rather than the code. The code also very short, is just a realize, the most important is your construction. A little code can have excellent effect sometimes.



## Forward propagation in a deep network

From input layer to output layer, the step is very simple, use the weight calculate the value, and then use the activations function, extract the useful feature into the next layer, and repeat this procedure.

The vectorizing Z[1] A[1] Z[2] A[2] is just the stacking of the z1 z2 and a1 a2, the calculate method is the same.

![image-20230910231929991](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910231929991.png)

View the whole procedure of these forward propagation, this is just a for loop, from layer 1 to layer L, do the same step, just calculate the avg by W, and then pass the feature to next layer through the activation function.

The procedure is similar to the single hidden layer neural network, but just repeat the step many times.





## Getting your matrix dimensions right

If you want your neural network has no bug, **you need think about the matrix dimension very carefully**.

When you debug, just use paper to write the dimension of your matrix.

Just as the 5 layer deep neural network, the w matrix dimension is decided to the previous layer and next layer. So the dimension of w[l] is (n[l], n[l-1]).

So we check whether the w[l] is equal (n[l], n[l-1]) , and the b[l] dimension need to be (n[l], 1), also, the dw and db's dimension is equal to w and b.

![image-20230910233225255](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910233225255.png)

If we vectorizing this procedure, the situation is as the same, just stack the previous vector into a big matrix.

Check the dimension about capital z, Z[l]'s dimension is [n[l], m].

![image-20230910234826125](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230910234826125.png)

These check make you know what's the dimension of these matrix, and make sure that all the matrices dimensions are consistent. This way will help you to eliminate some cause of possible bugs.





## Why deep representations?

why deep learning model is better than other simple models?



















