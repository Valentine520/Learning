# Hyperparameter tuning



## Tuning process

You have lots of hyperparameters in your training process. Different hyperparameters have different importance, the learning rate is most important you should tuning, and $\beta$ and number of hidden units and mini-batch size is sub-important.

<img src="C:\Users\13940\AppData\Roaming\Typora\typora-user-images\image-20230925081353017.png" alt="image-20230925081353017" style="zoom: 80%;" />

But how to tuning this hyperparameters?

If you just have 2 parameters, but their importance is not the same, if you use grid, your $\alpha$ maybe just 5 independent value, but if you use random value, you can have 25 independent values, you can find which one is the best value for the most important hyperparameter. So, **just try random values, don't use a grid.**

![image-20230925082103487](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925082103487.png)

Another tuning strategy is **Coarse to fine**. First, you may don't know what is the best value for your hyperparameters, but in your coarse sample of this entire square, you may get a good point, and around this point, the hyperparameter all prefer well.

So you can zoom in to a smaller region of the hyperparameters and then sample more density within this space.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925083058774.png" alt="image-20230925083058774" style="zoom:80%;" />





## Using an appropriate scale to pick hyperparameters

If your $\alpha$ is range 0.0001 to 1, you can divide them into 4 phrase, so you can just chose a random value to be the exponential parameter, you can get the random value from 0.0001 to 1. How to find the random range? you can just use $log_十range$, you can get the range. 

Like the 0.0001, you can get -4, and 1, you can get 0, so your sample range is from -4 to 0.

This random value is not linear, you can get the sample between 0.0001 and 0.001 as same as 0.001 to 0.01, this is not average random.

![image-20230925084407596](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925084407596.png)



One other tricky case is sampling the hyperparameter $\beta$.

The $\beta$ is from 0.9 to 0.999, just like you calculate the avg of 10 value and 1000 value. The importance about 0.9000 to 0.9005 and 0.999 to 0.9995 is not the same. If $\beta$ value more close to 1, the little change will cause huge effect.

So we use non-linear strategy to chose random value, make 0.9 to 0.99 and 0.99 to 0.999 has same sample point.

![image-20230925085515102](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925085515102.png)



## Hyperparameters tuning in practice: Pandas vs. Caviar

your data gradually change over the course of several months, you need **re-test hyperparameters occasionally** to make sure that you're still happy with the values you have.

About how to search a good hyperparameters there are two major schools of thought.

One way is if you **babysit one model**,  you have huge dataset but not a lot of computational resources, not a lot of CPUs and GPUs, you can basically afford to train only one model or a very small number of models at a time.

The babysit one model is to observe your model and change your parameters day by day, such like increase your learning rate if the result is better, and fill in your momentum, and so on. maybe on one day you found your learning rate was too big, you might go back to the previous day's model.

This approach can be called panda approach, because panda must care about just one children.

![image-20230925113043005](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925113043005.png)

Another way is **training many models in parallel**, use different parameters to train many different models, try a lot of hyperparameter settings and pick the one that works best. 

This approach is just like fish, have much more children, don't pay too much attention to any one of them but just see that hopefully one of them, or maybe a bunch of them.

![image-20230925113158193](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925113158193.png)

The way to choose between these two approaches is really a function of how much computational resources you have.

If you have lots of computer, you can just try Caviar approach, use lots of hyperparameters to see the result.



## Normalizing activations in a network

batch normalization makes your hyperparameter search problem much easier, makes your neural network much more robust to choice of hyperparameters.

Normalizing inputs can speed up learning, make the cost function more circle, then gradient descent more efficient.

![image-20230925114126612](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925114126612.png)

For the deep neural network, can we normalize the values of a, so as to train w, b faster. This is batch normalization dose. 

Rather than *a*, we prefer to normalize *z*.

![image-20230925114627897](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925114627897.png)

**Implementing batch norm.**

Just calculate the means and square of the parameters, then add a surplus elements $\epsilon$ to make normalization more stable, and then your means is 0 and variance is 1.

But sometimes we want make dataset distribute more widely and sparsely, so we want to change parameters means and variance. Use $\gamma$ and $\beta$ to control your parameters' means and variance.

![image-20230925115608600](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925115608600.png)

Maybe you don't want your data value accumulate at a little phase, so we can set a big variance, to make parameters sparsely and increase its means.

You can now make sure that your Z values have the range of values that you want.





## Fitting Batch Norm into a neural network

We know how to use batch normalization in a single layer, now we need to implement in deep neural network.

The process is very simple, just use the batch norm parameters in your network for every layer.

![image-20230925142227349](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925142227349.png)

The new parameters are $\beta$ and $\gamma$, the $\beta$ is not as same as $\beta$ of momentum, you would then use whether optimization you want to update the parameters of $\beta$ and $\gamma$.

Don't worry about this process, you might not end up needing to implement all these details yourself, you can just assist by TensorFlow built-in function.



**Working with mini-batches**

The parameters of b can just eliminate, because the normalization will eliminate the bias value.

So your parameters become w $\beta$ and $\gamma$, and its dimension is very important.

![image-20230925143755555](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925143755555.png)

Now you need modify your whole gradient descent process because of the new parameters.

Using programming framework will make using batch norm much easier.



## Why does Batch Norm work?

Let's consider a cat recognize program, your training data set is just black cat, but now you change your inputs, you want to predict the colorful cat, this is **covariate shift problem**.

![](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230925202615190.png)

How to implement covariate shift problem to a neural network?

The neural network want the output y-hat close to the ground true value y. The 3 layer unit, is aim to use previous value a1 a2 a3 a4 and find a way to map them to y_hat. 

If the input change, the previous layer will change all the time, so for the 3 layer(just a sample), the input value is changing all the time, and so it's suffering from the problem of covariate shift.

So the batch norm reduce the amount that the distribution of these hidden unit values shifts around.

![image-20230925202825305](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925202825305.png)

The batch norm will keep the means and variance although the dataset is changing, the different data cannot cause big effect of model, **make the neural network more stable and then has more firm ground to stand on.** 

(*There is an question about me, if the batch norm reduce the difference of different data, the features of different data can really been caught?*)

![image-20230925211231978](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925211231978.png)

**The advantage of Batch Norm is regularization.**

mean and variance has a little bit noise, just like dropout, it adds some noise to each hidden layer's activations, make the downstream hidden units not to rely too much on any one hidden unit, has a slight regularization effect.

![image-20230925212955957](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925212955957.png)

But don't think batch norm as regularization, just think the regularization is an almost unintended side effect.



## Batch Norm at test time

In your training step, your mini-batch may contain lots of training examples, so you can calculate the average and the variance of dataset, but for testing step, you may just need single example to test the neural network, not a mini-batch, so the means and variance cannot be calculated.

You can just use the training step's means and variance to calculate the exponentially weighted average, to get the final means and variance and calculate the normalization value.

![image-20230925235242675](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230925235242675.png)

Use batch normalization, you can train a deep neural network and get your learning algorithm to run much more quickly.



## Softmax regression

If you want classify not only single class but much more classes, you can use softmax regression.

You have 4 classes which you need to recognize, so the dimension of output y_hat is (4, 1).

![image-20230926075156874](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926075156874.png)

Finally, we need to get the probability about 4 classes, so we need to do some activation to the Z[L], let the final value can sum as 1, and stand for every classes probabilities.

In the following example, through softmax regression, we can get the probability of 4 classes, so we can judge what class this type is.

![image-20230926075950927](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926075950927.png)

The logistic regression use linear function to classify the class, but not 0 and 1, there more than 1 classes which you can use softmax activation to classfiy.

![image-20230926080810549](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926080810549.png)





## Training a softmax classifier

what is soft max? The hard max will map the biggest probability to 1, and other to 0, this is too hard.

But the soft max will use some soft method, you can see, every class has its own probability, you can just think the biggest is the most probably class.

![image-20230926081112862](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926081112862.png)

## How to training a soft max classifier?

#### Loss function

The loss function do is to look at **whatever is the ground truth class** in your training set, and it tries to **make the corresponding probability of that class as high as possible**.

There must single 1 and much more 0 in the truth sample, so we just focus the 1 element in the prediction.

![image-20230926081742124](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926081742124.png)

Let's see the forward propagation and backward.

Don't worry, not every time you need to get the whole formula, you just need to use deep learning framework to do this.

![image-20230926083130295](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926083130295.png)



## Deep Learning frameworks

It is not practical to implement everything yourself from scratch, there are many deep learning frameworks help you to implement these models.

There are lots of frameworks, but each of these frameworks has a **dedicated user and developer community**, chose which framework is just depending on your work. 

There also some criteria for you to chose your deep learning framework. How much you trust that the framework will remain open source for a long time.

![image-20230926083731607](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926083731607.png)



### TensorFlow

You can use tensorflow to find the W and b, which minimum the cost function J.

![image-20230926084140600](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926084140600.png)

Using tensorflow, you just need to complete your forward propagation but not spend much more time to calculate derivative about the backward propagation.

The powerful of tensorflow is you can just specify how to compute loss function, then it takes derivatives and just use some other optimizer with just pretty much one or two line of code.

![image-20230926103511873](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926103511873.png)

This is computation graph done.

![image-20230926104616645](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230926104616645.png)

