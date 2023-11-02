# Neural Network Basic

Explain how to train our model and you don't need to explicit use for loop. **(Vectorizing)**

Explain why there are forward propagation and backward propagation. (**Value and Derivative)**

We use logit regression to answer the question above.



## Binary Classification



![image-20230819064538409](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819064538409.png)

The image have 3 ways, so we just want to put them into a vector as a model's input, the x is 64* 64 * 3, the output is 0 or 1, this is a typical binary classification.



If we have much more than 1 input, it means we have m x vector, so **we need to organize them into a matrix**, the column is the training input sample, and also the column of y is the training label.

Now the training samples are associated with the column of the matrix, in the future we construct more complicated network, this will help us to locate the dataset.

![image-20230819065130484](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819065130484.png)



## Logistic Regression

The logistic regression is to use the discrete point x and the weight w and b to get the y, want to get a continuous function. 

The output y_hat is not in 0 and 1, because the wT * x + b, but the **sigmoid function** can help us do this. It can convert the input into (0,1), so the logistic regression can convert into the binary classification, the y_hat is between 0 and 1, and stand for the probability of the logistic regression.

![image-20230819071804406](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819071804406.png)



#### The sigmoid function

The function can force the input value into (0,1), so you can **just throw your output into a sigmoid function, you can get the probability**.

![image-20230819072136416](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819072136416.png)



## Loss and cost function

x(i) and y(i) ... all stand for the data associated the i-th example

![image-20230819072641864](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819072641864.png)

there is a question, how to quantity the cost about the continuous function's predict result?

![image-20230819072956644](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819072956644.png)



This is **loss function**: *to measure how good our output y-hat is when the true label is y*.

We can have 2-type of loss function about the above situation.

One is this, but this kind of loss function is no-convex optimization, so the gradient descent algorithm **will always get the local optimized result**. So we need to let the loss function become a **convex optimization problem.**

![image-20230819073050452](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819073050452.png)

So we have the new loss function: this loss function is similar to the GAN, this just want y_hat is approaches y, so when y is 0, the log(1-y_hat) need close to 0, so the y_hat approaches y, when y is 1, the y_hat need close to 1, so the log(y) need close to 0. 

![image-20230819073712972](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819073712972.png)

**The loss function measure one sample's cost, and the cost function measure the whole samples' cost.**

We can just plus all the loss value in the every sample.



## Logistic regression cost function prof

Why we use the above loss function?

The y_hat is a probability about the class equal to 1, so the 1-y_hat is the probability about class equal to 0.

So, if y = 1, the right probability is y_hat, if y = 0, the right probability is 1-y_hat.

![image-20230823154932395](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823154932395.png)

How to express this in a single equation?

![image-20230823155551735](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823155551735.png)

The equation can express the 2 situation about the probability.

Then, because we want to do gradient descent, we need find a function to which is convex. The log function, can be a suitable function.

![image-20230823155629591](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823155629591.png)

And then, in m examples, we need let all the probability multiple be the maximum. And then convert into let log m examples multiple maximum, the log can convert the continuous multiple become just sum of them. Just as follow:

![image-20230823160007108](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823160007108.png)

Use Maximum Likelihood Estimation, convert the sum of them become the average. Get the final equation.

![image-20230823160544648](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823160544648.png)

## Gradient Descent

How to train the model above? how to modify the parameters w and b, let the y_hat approaches y? First, you need know how become better, **find w, b that minimize *J(w, b)***

We can plot the 3d surface about the parameter w,b and the cost *J(w,b)*, the plot is convex, so we can find the minimal point is the bottom of this plot. We want to go down to the bottom, this is so call **Gradient Descent.**

![image-20230819074747963](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819074747963.png)

But the Descent is step by step, then go down to the minimal cost point.

![image-20230819075204727](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819075204727.png)



Let's ignore the parameter b, so we can get the 2-d curve, the J(w) is a simple function about w, and w is just a parameter. Just like y=f(x), you can just derivative, so you will get the slope, if you just go down as the slope, you will get to the bottom finally.

![image-20230819075558935](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819075558935.png)

The $\alpha$ is the learning rate, it decide the step this function to go down. If $\alpha$ is too small the model will cost much more time to optimized, and if $\alpha$ is too large, the model will not go to the bottom.

So **the $\alpha$ need to be a suitable value**, this is **hyperparameter**.

Further more, we discuss the situation about the whole cost function $J(w, b)$. We just also make w to minus some value which calculus by the partial derivative and b also do this repeatedly.

![image-20230819081101700](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230819081101700.png)



## Derivative

There are to much complicated math problem, so you just need know how to implement it but not all the principle, just apply these functions.

![image-20230820073246901](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230820073246901.png)

a increase 0.001 the f(a) increase 0.003 simultaneously. So the f(a) is 3 times of a, the slope of f(a) is 3, the derivative of f(a) is 3. **Use slope to understand the derivative.**

This little chapter is not very important or complicated for me, because I know the derivative, it's just want to express that in deep learning, **the derivative is very important even though necessarily.**



## Computation Graph

A complicated expression can use the new symbol to simplify, encapsulate little step such as b*c into a new variable b and finally we can express J as 3 * v.

This is computation graph, organize the computation step into a graph, this is forward propagation, from left to right, to calculate the value of the function.

![image-20230820074621585](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230820074621585.png)

#### Computing derivatives

We have encapsulate the every step function to simplify, if we want to get derivative, we just need **calculate every little step's derivative.**

Such as, if we want to know $dJ/dc$, **first we need calculate the $dJ/dv$, because $J$ is relate to $v$ directly**. The derivative $dJ/dc$ is 3.

![image-20230820080125805](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230820080125805.png)

Now, if we want to know  $dJ/da$, we can use the definition about the derivative:

when a change a bit, the v will change, and finally the j will change, so this is a propagation from a to j, and the derivative is how j influence v and how v influence a, so the total influence from j to v to a, is the derivative about j and a, this is **backward propagation.** 

![image-20230820082645178](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230820082645178.png)

**This is also called the chain rule in calculus.**

The  $dJ/da$ can transfer to calculate the  $dJ/dv$ and  $dv/da$, the a change will change v, and v will change J, so the chain propagation will finally change the J value, then we can get the derivative about J to a.

![image-20230820083129433](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230820083129433.png)

With the help of computing graph, we can just calculate the one step derivative, computer like do this thing, and can do it better.

Divide a big question into small continuously step, we can achieve it quickly.

If we implement this progress in python, we need think how to name the variable and the procedure. **We can just use $dvar$ to stand for the $dJ/dvar$ derivative.**

![image-20230821071753853](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821071753853.png)

So the $dw$ and $db$ is the brief notation of this derivative procedure. 

Because the **partial derivative is about only one variable**, so we can backward propagation this derivative procedure.

==Forward propagation calculate the value of formula, backward propagation calculate the derivative.==

One part change, will finally affect the result. 

And one more thing, if we calculate the $db$ we will get the $db=3c$ ,so we need to now the value of c, c there is not a variable, it's just a number, so **the initial of these parameters is very important.**

![image-20230821073140512](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821073140512.png)





## Logistic Regression Gradient Descent



### 1.Single training example

Recap the logistic regression task.

So we can construct the computing graph, **backward propagation can help us to calculate the derivative,**  the aim of this is to make L(a, y) less, so we need to gradient descent the w1 and w2 and b parameters.

![image-20230821074254869](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821074254869.png)

We need to calculate the derivative step by step, first the $dL(a,y)/da$ and then $da/dz$ ... 

This is the derivative about single sample x1 and x2, the result of $dw$ is related to x1 and x2, the sample point.

![image-20230821075017156](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821075017156.png)



### 2. m training example

If we just have single training example, the Cost function is as same as the Loss function, but if we have more than one training example, **the Cost function is the average of the Loss function in m training examples.**

So we need know **the derivative of the Cost function, it just the average of the loss function's derivative in m examples.**

![image-20230821075501440](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821075501440.png)

We just need to calculate the average example to get the derivative, such as the $dw=x1(a-y)$ so the x1 **need to be the average example of m x1.** Just as follow:

![](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230821080212001.png)

So in this step, there are the derivative cumulation, so the *pytorch* will do this cumulation, aim to help us calculate the derivative.

Then use $dw / m$, to calculate the average value of the derivative.

![image-20230821080513058](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821080513058.png)

Calculate the derivative, then use the learning rate to do gradient descent, update the parameters.

There is a problem, we use for loop two much, it will cause the training procedure very slowly, we not able to deal with the large datasets.

![image-20230821080709509](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230821080709509.png)

**So try your best to avoid using for loop explicitly,** then you can deal with the larger datasets and millions of parameters.

This technic is called **Vectorization.** It will help us getting rid of for loops, accumulate the training speed.



## Vectorizing

A series data can organized into a matrix, and then use matrix computation to calculate the original expression. 

![image-20230823112726618](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimageimage-20230823112726618.png)



#### **Vectorized will speed calculation.** 

The time can use tic and toc to calculate, the for loop waste much time to execute the loop, the speed is very slow, but the matrix computation can finish fast.

![image-20230823140734910](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823140734910.png)

This is matrix computation.

![image-20230823140748056](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823140748056.png)

Because CPU and GPU all have **SIMD(single instruction multiple data)** property, so the computer will optimize our computation, such as np.dot(A, B), this will more efficient than just use for loop.

The python numpy will apply the function into every parameter in the ndarray, such as exp function, exp(ndarray) will exp every parameter in the ndarray, its convenient and efficiency.

![image-20230823112639841](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230823112639841.png)



## Vectorizing Logistic Regression



### How to eliminate for loops in logistic regression?

We can use matrix multiple to calculate single example z and a, then we can organize all of examples into a matrix, Z is a vector of z1 z2 and so on.

So the final equation is $Z=np.dot(W.T, X) + [b, b, b....]$, the python's character will let b be the [b, b, b...] automatically.

![image-20230823112550944](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823112550944.png)



### Vectorizing Logistic Regression's Gradient Computation

Also, the derivative about every parameters can merge into a matrix like $dW = [dw1, dw2, dw3...]$, so the inner for loop can be placed of $dW += Xdz$, the X is a example.

![image-20230823144154043](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823144154043.png)



And for more than one example, we need eliminate the explicit for loop, just let example into a X matrix.

![image-20230823144038730](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823144038730.png)

Also for $dz$, let $dZ=[dz1, dz2...]$ and A Y also do this.

The aim of the organization is to make all example into a matrix, then use $XdZ$ to calculate the gradient for every example, then calculate the average, we can get the final gradient descent value.



![image-20230822073530784](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230822073530784.png)

The w and b update procedure can describe as a simple cumulation. 

![image-20230822073737140](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230822073737140.png)

Vectorize this procedure, we can get the follow equation.  The $dw$ is a vector, stand for every weight's derivative, like w1 is equal to $average(x1dz1 + x2dz2 + x3dz3 ....)$

![image-20230822074041057](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230822074041057.png)

### Implementing Logistic Regression Vectorizing

Left is the two for loop version, too slow, and right is the vectorized version, which can run faster.

The J is loss function, its aim is to calculate gradient, so we can just calculate and ignore it such as the right version.

![image-20230823150106334](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823150106334.png)



## Broadcast in Python

There are 4 kinds of food, we want to calculate the percentage of every food's Carb 、Protein and Fat about the total calories.

![image-20230823150642058](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823150642058.png)

One way we can do is to calculate the total calories and organize into a vector, and make every column to divide the related total calories value.

But assistant by Python's broadcast mechanism, we can just use $A / A.sum(axis=0)$, (3,4) / (1, 4) will be broadcast into (3, 4) / (3, 4), the 3 rows are replicas of 1 row.

![image-20230823151105919](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823151105919.png)

The b in the above will also be extend to [b, b, b,  xxxx] just for calculation convenient.

The A.sum(axis=0) is hard to understand, but you should remember the follow graph, axis=0 means calcum as row, and axis=1 means calsum as column.

![image-20230823151613847](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823151613847.png)

### more broadcasting examples

Just like b -> [b, b, b...]

![image-20230823152344698](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823152344698.png)

The follow example will also copy the row of the first line, let them can calculate directly.

![image-20230823152527249](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823152527249.png)

Also, copy the first column to be (m, n) shape for plus directly.

![image-20230823152732041](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823152732041.png)

#### General Principle 

![image-20230823152933399](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823152933399.png)

and more complicated, if a row plus a column, all of them will extend at the same dimension.

**The python broadcast can make you done with fewer lines of code.**





## Some tricks on python/numpy vectors

python is very useful but can make mistakes easily also. There are some little tips which will help you reduce your mistake in programming:

#### **Do not use rank 1 arrays**

In the (5, ) shape a, its rank is 1, it not a row vector nor a column vector, it like a python list. And very weird, the dot operation and so on in the rank 1 'vector' is also unknown.

![image-20230823153722125](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823153722125.png)

So, the best way is avoid using this kind of arrays,  just let an array be (1, n) or (n, 1), be explicit a row or column vector.

![image-20230823154128302](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230823154128302.png)

