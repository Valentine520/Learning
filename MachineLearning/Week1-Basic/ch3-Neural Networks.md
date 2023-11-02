# Neural Networks



## Overview

The training examples x1 x2 and x3 and so on, as the input of the model.

And then calculate the z use input x1 x2 and x3, get z1 z2 and z3, the z[1] is the first layer of the model, and we need to calculate the z[2] use z[1]'s three value.

**Attention:**

*The x(1) stand for the 1 training example, and the x[1] stand for the first layer of network.*

![image-20230824104623484](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824104623484.png)



Transfer two much training examples into single z[2], and then use sigmoid function to calculate the final value y_hat. 

The first layer's different node has different weight, use same input x1 x2 and x3, calculate every layer one node, and then calculate the layer 2, use the different weight also.

The forward propagation as follow:

![image-20230824105850139](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824105850139.png)

And the backward propagation can describe as follow:

![image-20230824105939467](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824105939467.png)

Just as same as the original backward propagation, just a little bit different, is the layer. Layer[2] derivative and then layer[1] derivative, the method is all the same.



## Neural Network Representation

Explain what the network we drawing represent.

The **input layer** is our training examples, and **output layer** is our final result.

The **Hidden layer** means that the value of them is not visualize, **we don't know the value of them,**  we just know the input layer's training example and output layer's model result, but we care about the hidden layer.

![image-20230824110512276](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824110512276.png)



**The X vector can use a[0] to represent**, it means **activations,** refers to the values of different layers of network are **passing** the passes layers. (This is what activations means)

![image-20230824110828216](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824110828216.png)

And also, the value will pass from input to output, so there are 3 activations layer, we use a[0], a[1], a[2] to represent.

![image-20230824111251416](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824111251416.png)

**The above Neural Network is 2 layer NN**, because the input layer always the 0th layer, so we actually have 2 layer, the 1 hidden layer, the output layer.



Our network has some associated parameters, like layer 1 has w[1] and b[1], the dimension is [3, 4] and [4, 1] this is decided by the dimension of network layer.

The a[1] layer's a1 a2 a3 and a4 can have different weight about the input x1 x2 x3, so the dimension is [4, x], and the a1 is equal to x1w1 + x2w2 + x3w3, so the full dimension is [4, 3].

Dimensions is very important in our python code.

![image-20230824111641203](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230824111641203.png)



## Computing a Neural Network's Output

Our logistic regression's computing is follow:

![image-20230830071143136](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830071143136.png)

If we have neural network, how to compute now?

![image-20230830071220249](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830071220249.png)

In the hidden layer, the every node is calculate the value of sample, get the y1 and y2 y3 y4 and so on, this is the first layer.

**Every hidden layer node calculate as the logistic regression two step.**

![image-20230830071425875](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830071425875.png)

We can organize them into a new equation :

![image-20230830071511283](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830071511283.png)

Every node has the relative parameters w[1] and b[1], in the first layer. Execute the two step calculation and get the a value for every node.

Make the equation vectorizing, we can get the matrix. This is why the first layer parameter w's dimension is 4 x 3, and the b[1] is 4 x 1.

![image-20230830072026117](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830072026117.png)

Then a[1] is equal to sigmoid z[1], numpy will apply this operation to every single value of z[1].

![image-20230830072143668](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830072143668.png)

Finally, the second layer is calculate the avg about every a result in layer 1.

All in all, if you want to compute the two layer neural network, you just need to implement these four equation, the 1 2 equation compute the first layer, and the 3 4 equation compute the second layer.

![image-20230830073118924](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230830073118924.png)

Above computing process is just about single example, but we always have much more than one example.

Stacking training example into a matrix is also useful, this neural network we use is also vectorized. This approach can computing all the training example at the same time.

There is a mistake, the x1 x2 and x3 just single training example, but the neural network need three parameters so the final y_hat is just **computing by single training example.**

Let we go to the detail of vectorizing about multiple examples.



## Vectorizing across multiple examples

If we have multiple training example,  we need calculate y_hat(1) and (2)... for every training example.

![image-20230907064338181](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907064338181.png)

We can just use for loop to calculate every y_hat.

![image-20230907064513630](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907064513630.png)

We want to vectorizing this whole process.

Just stacking the training example x(1) and x(2)... to get a big matrix X, so we can just use the original W[1] to calculate the z(1) z(2) then stacking into a matrix Z[1], and then calculate the second layer.

![image-20230907065126419](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907065126419.png)

This matrix left corner value, is related to the first training example and the first node of hidden layer. we can get the following values.

![image-20230907065230321](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907065230321.png)

The horizontal of A matrix scanning of different training examples, and the vertical is about the different hidden unit.

in order  understand this vectorizing, we can just split the whole process into every single step, just ignore the parameter b.

![image-20230907070302668](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907070302668.png)

If we want to get the whole result, we can just use the python broadcast mechanism to plus b[1].

We can get the final equations.

![image-20230907070439906](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907070439906.png)

Until now, we have already realize the neural network and calculate the y_hat by sigmoid function, but this activation function will now always appropriate.





## Activation functions

The activation functions is not to activate something, but indicate that how to pass the activation node features to the next layer, **it just means reflect the neural unit input features to the output and pass this features to next layer**.

In the linear regression problem, our neural network use the sigmoid function as activation function.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907071550257.png" alt="image-20230907071550257" style="zoom: 67%;" />



We can have more general function just as **g(z[1])** , just as follow:

![image-20230907071758726](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907071758726.png)

tanh(z[1]) is actually the translate of sigmoid function, its average value is 0, split 2 type as negative and positive. 

**using a tanh instead of a sigmoid function kind of has the effect of centering our data.**

make the mean of data is 0 but not 0.5.

Every activation function has its usage, the tanh is better than sigmoid function in almost situation. But if our training task is a 2 type classification, the sigmoid function is very useful.

So we can use different activation function for different layer play to their strengths.

In the hidden layer, we use tanh function and in the output layer, we use sigmoid function for this 2-type classification task.

![image-20230907072624694](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907072624694.png)

Also, we usually use the ReLU activation function, this function is very simple, just a = max(0, z), can derivative very quickly.

![image-20230907074016958](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907074016958.png)

This function is not derivative in (0, 0), and as the 0, this node will not activate, because its derivative is 0. The training process is better than sigmoid and tanh function.

To improve the ReLU, there is Leaking ReLU, but we seldom use it. We always use just ReLU.

There are all kinds of activation function, we can just chose one suitable. But we prefer tanh to sigmoid, and always use ReLU function.

![image-20230907074611123](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907074611123.png)



There are many different kinds of choice in deep learning, about initialize the weights and activation functions and neural network unit.

Your choice must suitable your application, must know what parameter is more efficient.



## Why do you need non-linear activation function?

why we must use the activation function?

If we just has the identity activation such as g(z)=z, this is linear activation function, as below.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907110952166.png" alt="image-20230907110952166" style="zoom:67%;" />

We will finally get the simply equation:

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907111302883.png" alt="image-20230907111302883" style="zoom:80%;" />

The hidden layer is not usable, it just change the weight W, and b, the neural network become a single layer network, just a linear composition.

If we use the linear activation function in the hidden layer and the sigmoid function in the output layer, the network is equal to single layer logistic regression.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907111946776.png" alt="image-20230907111946776" style="zoom: 80%;" />

So the hidden layer cannot use linear activation function,  but the output layer can use this linear function, this is related to your training task.

**Use non-linear activation function is critical part of neural networks.**



## Derivatives of activation functions

The gradient descent method need to calculate derivatives to these activation functions, so we can just use the slope to calculate the derivatives.

### Sigmoid activation function

We can just use calculus to compute the slope of this activation function, the derivative is g(z)(1-g(z)), so if z is positive, the slope is approaching 0, and z is negative, the slope is approaching 0 too.

![image-20230907112823228](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907112823228.png)



### Tanh activation function

![image-20230907113310904](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907113310904.png)



### ReLU and Leaky ReLU

The derivative of g(z) if z qual to 0 is undefined, but we can just let the derivative equal to 0, z = 0 will not always happen, so we don't need to care about this.

![image-20230907113533292](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907113533292.png)



## Gradient descent for neural networks

The whole training process is first calculate the cost function, and repeat the derivative dw and db ... Then update the parameters such as w and b, this is gradient descent.

![image-20230907154357427](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907154357427.png)



**Formulas for computing derivatives:**

The forward propagation is well to understand, but the back propagation is hard to understand.

![image-20230907155005826](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907155005826.png)

The np.sum(keepdims=True) is avoid the python generate the weird rank 1 matrix, we need the final result is (n, 1) but not a rank 1 vector (n). Or we can use reshape to make matrix as our need shape.

 

How to get the derivative above? help by computing gradients.

Said briefly, the derivative process is just chain rule. calculate da first, da is partial derivative of the loss function. and then calculate dz, the sigmoid function. The backward propagation is that we must use the before derivative to calculate the following derivative.

![image-20230907160043406](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907160043406.png)

Then for our 2-layer neural network, the compute graph is as same as above, but little bit different. 

The gradient descent step is also the same, just have an extra layer, more derivative chain.

![image-20230907164733233](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907164733233.png)

The dz[1] is the every element product.

![image-20230907164833014](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907164833014.png)

If you think something hard to understand in backward propagation, just think about the matrix's dimensions, such as dW[2] 

The dw[1] and dw[2] is very similar, in fact, the forward propagation is similar, so the backward propagation just has little bit difference about the x.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907170451530.png" alt="image-20230907170451530" style="zoom: 80%;" />

The final step is to vectorizing this whole process, because of **the matrix can calculate as single block,** so just stack the value of multiple examples.

The backward propagation's vectorizing is depending on the forward propagation equations.

![image-20230907171021955](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907171021955.png)

Don't worry about this derivation process, we can just know what this come from, but don't need to derivation from 0 to 1, every time, just use it.



## Random Initialization

The parameters initial value is very important, this is related to the following gradient descent.

**What happens if you initialize weights to 0?**

![image-20230907172307943](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907172307943.png)

The hidden layer's 2 node calculate as the same, so it's update is all the same, this situation will cause there are 1 layer and 1 node actually in this neural network, the network will become very bad.

even though you have deep layer and much more unit in one layer, your network is still a simple network.

The solution to this is to initialize your parameters randomly, let every node learning different thing, this is a good network. Break the symmetry problem, let W is a random weight, but the b can be the 0.

**Our W[1] need to be very small**, to make Z[1] small, because our sigmoid function's derivative, if our Z[1] is too large, the training process will cost too much time and the result will not very suitable, the model will fit slowly.

![image-20230907173308604](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230907173308604.png)

There are more good constant number for W[1], 0.01 is one of them, if you training a very deep neural network, you need chose a new constant value. This is the hyper parameters.



**Loss function is the value of the module, but the gradient descent is to calculate the derivative by the forward propagation, and update the parameters to make loss function smaller.**

The gradient descent calculate is just across by the forward propagation,  and calculate the derivative.





