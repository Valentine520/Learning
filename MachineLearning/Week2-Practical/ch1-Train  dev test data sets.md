## Train / dev /test data sets

**Applied ML is a highly iterative process**, we need to try and make our model be better and better.

ML has been used in many scenarios, such as NLP, Vision, Speak, Structured data(Ads, Search engine, logistic).

![image-20230912193131238](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230912193131238.png)

Go around this cycle many times and find a good choice of network hopefully. How efficient to iterator will decide how fast you can train your model. The train/dev/test sets is very important for this efficient.

### ***Data sets partition***

The original data can be divided into 3 types, the training set, the core, and the hold-out cross validation set, or the development set, just call this dev set. final portion is test sets.

![image-20230912195007897](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230912195007897.png)

We use the training sets to train our models and across the different models performs best on our dev set, then we can just estimate the model in our test data sets.

The preview rate is 70% training and 30% testing or 60/20/20 organization. 

But when the dataset become more and more huge, the percentage of these partition will become different.

We need 1million data to train our model, to fit all kinds of situation, but if you want to validation which algorithm is best, there are just 10 kinds of algorithm you need to estimate, so you just need 1 thousand validation data, and 1 thousand test data. The percentage become 98/1/1.





### *Make sure dev and test sets come from same distribution*

With the developing of the dataset's scale, the training set and dev sets maybe come from different resource, just like your training data is come from the webpages beautiful images but the user's update their test dataset which maybe a blurry picture.

![image-20230912213736486](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230912213736486.png)

### *Not having a test set might be okay*

The goal of test set is to give you a unbiased estimate of the performance of your final network, of the network that you selected.

If we don't need the biased estimate, we can just train different models in different training set, and evaluate them on the dev set, and use that to iterate and try to get to a good model.

But people often call this validation set as test set. It's aim is similar, just want to find the best model, you need know it's principle, that the set is validation set not the test set.





## Bias and Variance

In deep learning, we often talk about the bias and variance separately, but we talk little about the bias-variance trade-off.

The underfitting is the function can't describe the data sets very well, and the overfitting is the function that fit the data so well, but it maybe cannot be suitable to general data sets.

We need the just right classifier

![image-20230912215424810](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230912215424810.png)

The Bias indicate that the training fitting, just that your train set error. But the variance is about the dev or test fitting, the dev set error .

So, if your train set error is large, this is so call high bias, if your dev set error is large, there must be high variance.

If you have large train set error and also have larger dev set error, this means you have high bias and high variance.

![](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913001729471.png)

All of this is related to your real correct ability, just like our eyes accuracy is almost 100%, so the train set error 1% and dev set error 11% is high variance.

If the optimized error is 15%, just say our eyes has 85% accuracy, the train set error 15% and dev error 16% is acceptable.

The high bias and variance is about the not fit and over fitting, just like follow:

![image-20230913003230280](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913003230280.png)

The high bias is because the linear classifier cannot fit the data set well, and high variance 

because the classifier too flexibility to fit those two mislabel, this model is a little bit contrived.



## Basic recipe for machine learning

We will meet many kinds of situation about the training process or the testing process. 

If your model has high bias, you must consider your training data preform, you can change your network, select a bigger network or train longer. Any way, you need to optimized your network, the structure or the activation function and so on, until you solve this bias problem.

![image-20230913064814652](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913064814652.png)

But you may counter the data over fitting problem, this is high variance. You can just use more data or Regularization or change your neural network structure.

The final logistic process is as follow:

![image-20230913064952644](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913064952644.png)

You must locate your model problem in training process, if your model has high bias, more and more data is not efficient. You must figure out is high bias or high variance.

In the early era of deep learning, we often talk about the bias and variance trade off, because we can not reduce bias or variance separately, if we minus our bias, the variance maybe increasing.

![image-20230913065511250](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913065511250.png)

But with the development of deep learning, you can just train a bigger network continuously, use larger data, chose a appropriate regularization function. The modern tools can just drive down bias and not effect the variance.

So there's much less of tradeoff where you have to carefully balance bias and variance.



## L2 Regularization

If your training data over fitting, also the high variance, you can just use more data or the **regularization** operation. Regularization is very useful to solve the over fitting problem.

![image-20230913070329152](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913070329152.png)



The formula is very simple, the regularization is just like the linear algebra's regularization.

![image-20230913070747322](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913070747322.png)

The W is a high dimension variable, it contains the value of b, so you don't need to plus a b as regularization, we just need to compute the **L2 regularization** of W. L2 regularization is the most convenience approach.

You can also use the different regularization method, such as L1 and L2. But the w will end up being sparse, it means the w vector will have a lot of zeros. Some people said this method can compress the model because of the 0, you need less memory to store the model. But this compressing is not L1 regularization purpose.  

![image-20230913071040136](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913071040136.png)

Now people trend to use rather L2 regularization than L1 regularization.

![img](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imagev2-f8cf4f5d21042f66e5ecc0c4f1c415cb_r.jpg)

The $\lambda$ is regularization parameters. Consider the terms of trading off between doing well in your training set versus and also set that normal of your parameters to be small, the $\lambda$ need to be small to avoid over fitting.

The $\lambda$ is also an important hyper parameters we need to chose, because of the python key words lambda, so we use lambd to indicate the parameter of $\lambda$.



### *How to implement Regularization in neural network?*

Let's start at the Cost function. Because of the regularization, we need to plus a surplus value, this value can compute through the sum of every layer's weight matrix.

The dimension of W is (n[l-1], n[l]), so we need do the two for loops, compute every layer's L2 normal value.

This norm is called the **Frobenius  norm** of the matrix, denoted with a F in the subscript. This is not called the "L2 norm" but the Frobenius norm.

![image-20230913101027098](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913101027098.png)

We realize the forward propagation procedure, but how can we do gradient descent during backward propagation?

We calculate the $dw$ before from backward propagation, and because of this regularization, now the $dw$ must add the extra regularization terms at the end, $\frac{\lambda} {m} * w^l$.

For this reason, the L2 norm often be called as the ***Weight Decay*** norm. You can see the final equation of this, W first multiple a fraction value, decay, and then subtract the previous derivative $\alpha * dw$. The W become smaller and smaller, this can prevent the data over fitting.

![image-20230913103214437](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913103214437.png)





## How regularization prevents over-fitting

You should know how the over-fitting looks like. In the previous learning, we know if your training set error is very small, but your validation error is very large, your model is over-fitting, because your model learn every detail of your training set, but not general characteristic about this kind of data, so the validation set cannot fit this model well.

In the neural network view, if we have deep neural network, and our neural network every layer contains too much units, this can cause  the over-fitting. What we can do is just make the model become simple, less complicated, to make it suitable to every general data set.

![image-20230913113719351](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230913113719351.png)

If your $\lambda$ is too large, the weight of this neural network will approximate to 0. The before W matrix become a sparse matrix, many hidden layer unit become 0, unactive, so the complicated deep neural network become a simple neural network.

The larger $\lambda$ , the simpler the network, the neural network has trend to transfer high variance to high bias, just become a simple logistic regression, not a complicated neural network.

We can use a activation function tanh to explain this reason about why regularization can solve the over-fitting problem.

![image-20230914110457601](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914110457601.png)

Because the loss function is  $\lambda$ * $W_F$, so if we want decrease the loss of this model, we must decrease $W_F$, make W a sparse matrix, so the $W_F$ can decrease. 

With the W decrease, the Z will be relatively small, the tanh activation function will be almost a linear activation function, if every layer all the linear function, the deep neural network will become a simple network, so can solve the over-fitting of this model.

We add the regularization is aim to make W smaller, so our model can be trained quickly and avoid over-fitting.



## Dropout Regularization

Recap the ReLU activation function, if value is negative, the weight is 0, so eliminate some unit, make network simple and can be trained quickly.

The Dropout Regularization is as the same, every layer every unit have a probability about whether select, or whether dropout. Like the right graph, some unit have been dropout, so the network become simple, solving the over-fitting problem.

![image-20230914111200664](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914111200664.png)

**How to implement dropout regularization in python code?**

Just use the boolen matrix, use np.random.rand(shape) < limit, to generate a boolen value matrix, indicate every unit's selected or unselected, reduce the complexity of neural network. 

Then use the d3 boolen matrix to update the activation function, update a3.

![image-20230914163200804](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914163200804.png)

Finally, in order to not reduce the expected value of z4, the a3 need to divide the keep-prob, eliminate the effective of these dropout unit.

![image-20230914163628470](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914163628470.png)

In the test time, because of the unit drop out is randomly, so we just use no drop out to test our model, this is different between training time.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914164026545.png" alt="image-20230914164026545" style="zoom:80%;" />



## Understanding Dropout

Because every iteration just reduce the scale of the neural network, so can we just use a simple and small neural network to instead of regularization? No

Intuition: Can't rely on any one feature, so have to spread out weights. Every layer input will be eliminate randomly, so we cannot rely on just one feature, these feature's weight cannot be large, because maybe the large weight unit will be eliminate. So, like L2 regularization, the drop out regularization will shrink the weights, prevents over-fitting.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914170853992.png" alt="image-20230914170853992" style="zoom: 80%;" />

Every layer has  different scale of unit, for the complicated layer, you can set keep_prob value to be smaller to apply a more powerful form of dropout, for the simple layer, you can just set the keep_prob value equal to 1, it means keep all units.

![image-20230914171942451](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914171942451.png)

You should remember, the drop out method is a regularization technique, it helps prevent over-fitting, you shouldn't use dropout unless your neural network is over-fitting.

But the disadvantage of dropout is cannot debugging the result, because of randomly, we cannot compute the previous loss function value again.



## Other Regularization Techniques

### Data augmentation

Training a neural network may need lots of data set, if we just have some data, we can use data augmentation to expand our dataset.

Computer sometimes not very smart, you can just do something little bit change, then you can get a new data.

Just as follow, we can flip the original image to get a new image, and also crop the image randomly.

These fake data maybe have many redundant, so the training effect is not as well as the true dataset. But the cost of these fake data generation is free, we can use them to improve our neural network model.



![image-20230914172913224](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230914172913224.png)

### Early stopping

Your neural network training step may decay the loss continuously, but our aim is to make the model can deal with the task well, but not the loss function.

So in the training step, our model's dev set error will decrease first, with the training process, the dev set error may increase, this is over-fitting. Our model is over-fitting by the training set, there are too much training set training step but ignore the general value, it focus on the training data set.

So, we need to **early stopping**, to stop the training on your neural network halfway, to prevent the increase err of dev set.

![image-20230915081506241](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915081506241.png)



The main downside of early stopping is that this couples, these two tasks.

There are two main task in our deep learning, first is Optimize Cost Function J, we can use gradient descent or Adam to optimize it. The second is not overfit, we can use regularization and data augmentation to realize it. This is called Orthogonalization.

But the early stopping cannot do these problems separately, if you not optimize the cost function, you stop deal with over-fitting.

![image-20230915081653097](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915081653097.png)

If you use L2 regularization to prevent over-fitting, you may cost much more time to train you model, and you don't know what is the best parameter of $\lambda$. **The early stopping can just do once gradient descent.** This is early stopping advantage. Every technique has its own benefit.

But we prefer to use L2 regularization often, just try different $\lambda$ parameter.



## Normalizing inputs

When training a neural network, one of the techniques that will speed up your training is normalizing your inputs.

**The first step is to subtract out or to zero out the mean**, move the training set until it has 0 mean.

![image-20230915102603457](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915102603457.png)

The second step is to normalize variance, every input variable has its own variance, we need to make them similar. The variance of x1 and x2 all become to 1.

Attention, we want our testing set and dev set do the normalizing as same, so **we need use our training set inputs to normalizing test and dev sets.**

![image-20230915102902367](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915102902367.png)

### *Why normalize inputs?*

Your inputs x1 and x2 has a big different range, x1 maybe from 0 to 1, and x2 from 1 to 1000, so the weight w1 and w2 may huge different. The image like a bowl, if you use gradient descent, you need repeatedly the descent, from bottom to top and your iteration will very long.

If you normalize the inputs, the w1 and w2 may very similar, your loss function just like more symmetric, so you can just do gradient descent from every where, your step can set a little big, your training step maybe very fast.

![image-20230915104845273](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915104845273.png)

The range of your input should be similar, so that your training step can be very quickly. just like x1 and x2 and x3, range very small. If there a x4, range from 1 to 10000, that really hurts your optimization algorithm.

![image-20230915105650459](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915105650459.png)

If your features came in on similar scales, then the normalize step is less important.



## Vanishing/exploding gradients

Consider you have a deep neural network, every single layer has just 2 unit, if the weight of W is more than identity matrix I, the final y_hat value will increase exploded, but if W is less than identity matrix, the y_hat value will become very small, just vanishing.

It will take a long time for gradient descent to learn anything.

In fact, for a long time this problem was a huge barrier to training deep neural networks, but there isn't property solution to solve this problem completely, what we can do is just **chose your initialize weights carefully.**

![image-20230915161515366](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915161515366.png)



## Weight initialization for deep networks

### Single neuron example

The larger n is, the smaller $w_i$.

![image-20230915162224494](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915162224494.png)

If your activation function is sigmoid or tanh, you can use sqrt(1/n[l-1]), if your activation function is ReLU, you can use sqrt(2/n[l-1]).

This is just recommendation, you can just use your own method to make your network more optimized.

Hopefully that makes your weights, not explode too quickly and not decay to zero too quickly.

![image-20230915162912132](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915162912132.png)



## Numerical approximation of gradients

You should check your training step to keep them correctly, during backward propagation you should check your derivative whether right or false. you should **checking your derivative computation.**

If your function is just $f(\theta)=\theta^3$, you can use two sided difference way of approximating the derivative, this is just the definition of calculus, but not the limit, we chose a just mall enough distance. 

The value of derivative may be different but little bit, so we can use this numerical approximation of gradients to check your formula gradient descent. 

 ![image-20230915163944354](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915163944354.png)

And we use the two sided difference way of this approximating, this is much more accurate than single difference way. We often use the left method to go gradient checking.

![image-20230915164729961](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230915164729961.png)



## Gradient Checking

Because our gradient checking above is about single variable $\theta$, but in the neural network, the loss function J is related to many parameters, so we need take these parameters reshape into a big vector $\theta$.

![image-20230916063225708](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230916063225708.png)

The J maybe have multiple $\theta$, because of the different examples, so we use for each i, to iterator the gradient checking for every example.

Use approximate function to compute the $d\theta_a$, and compute the distance between the gradient descent derivative $d\theta$, if the distance is bigger than $\epsilon$, you should attention to your gradient result, be careful, avoid making bugs.

![image-20230916063701011](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230916063701011.png)



## Gradient Checking implementation notes

There are many efficient tips for you during training your neural network.

### 1. Don't use in training - only to debug

In the training process, if you compute the derivative every step, the speed will be very slow, so just compute derivative if you want to debug.



### 2.If algorithm fails grad check, look at components to try to identity bug.

If you find that is $d\theta_a$ is very far from $d\theta$, what I would do is look at the different values of i to see which are the values of i, that make the $d\theta_a$ are really very different than the values of $d\theta$.

The $d\theta$ is related to the $db_i$ and $dw_i$, so you can just check your neural network parameters.



### 3. Remember regularization

![image-20230916072108731](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230916072108731.png)



### 4. Doesn't work with dropout

Because of the dropout technique can just eliminate unit randomly, so if we do the double check about the gradient, it's very difficult.

So we avoid gradient checking and dropout do simultaneously, and we prefer the double check than dropout, you need to be correct first and then be effective.



### 5. Run at random initialization; perhaps again after some training.

Your parameters may have something wrong to compute the derivative, if them trend to 0, so you can just do gradient check in the parameters initialization and train your network, and check the parameters again.





## Experiment

### Initialization

The weights 𝑊[𝑙] should be initialized randomly to break symmetry.

Initializing weights to very large random values does not work well. - Hopefully initializing with small random values does better. 

If you initialization by multiply sqrt(2/ dimension of the previous layer), the training model can be better.

![](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230917074228586.png)



![image-20230917074356637](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230917074356637.png)

![image-20230917074404709](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230917074404709.png)

This is very important to set a suitable value for initialization.

**Remember:** Different initializations lead to different results - Random initialization is used to break symmetry and make sure different hidden units can learn different things - Don't intialize to values that are too large - He initialization works well for networks with ReLU activations.



### Regularization

The drop_out technique just use the situation of model over-fitting, so you must use is in the layer which has much more units, to prevent over-fitting, if your layer just has two or three units, it's not necessary to use drop_out regularization. Don't worry about all units be eliminated, just use it in the layer which has much more units.

Dropout is a regularization technique. - **You only use dropout during training**. Don't use dropout (randomly eliminate nodes) during test time. - Apply dropout both during forward and backward propagation. - During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.



### Gradient Checking

Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation). - Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.
