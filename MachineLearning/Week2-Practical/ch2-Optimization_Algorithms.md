# Optimization Algorithms

Applying machine learning is a highly empirical process, is a highly iterative process. You should train a lot of models to find one that works very well.

A good optimization algorithm can speed your training process.



## Mini-batch gradient descent

Vectorization allows you to efficiently compute on m examples.

![image-20230919001509565](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919001509565.png)

But if your m is 5,000,000, your training process will be very slow, you need calculate entire dataset gradient, then you take one little step of gradient descent.

It's difficult because we need to calculate the entire dataset, but we can just split training set into smaller, little baby training set, these baby training sets are called mini-batches.

If your m is 5,000,000, it's too large, we can use mini-batch of 1,000 each, let the 1000 examples as a mini-batch to train the model. You just need calculate derivative for every whole batch but not all the datasets.

The symbol about the mini-batch is {t}, like the unit (i) and layer [l].

![image-20230919160939502](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919160939502.png)

Then you have 5,000 mini-batch, so you need a for loop, to train every mini-batch step by step.

![image-20230919162334853](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919162334853.png)

This process is exactly as same as we use before, just split the whole dataset into small batches, and we can train every epoch to improve our model.



**Why prefer mini-batch gradient descent but not the direct gradient descent?**

Because your machine has **limited memory**, you cannot deal with the huge dataset at the same time, so you just need to split them into a small batch, make your machine can compute quickly.

And if you just calculate the once gradient, you may deep into the local optimized, if you use mini-batch gradient descent, you can avoid this situation, you can calculate multiple loss function and see more darkness.



## Understanding mini-batch gradient descent

In the batch gradient descent, your function cost is a function about iterations, and must decrease. In the Mini-batch gradient descent, because your cost is about every single batch, so it may has many undulations(波动), because previous parameters maybe not suitable for the next batch examples, but the cost function is also trend to be lowest.

![image-20230919163857761](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919163857761.png)

In the mini-batch gradient descent, one of your most important task is to chose your mini-batch size.

If you chose mini-batch size equal to m, it's actually the batch gradient descent, if you chose your mini-batch size equal to 1, you can get the stochastic gradient descent, every example is it own mini-batch.

![image-20230919164429801](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919164429801.png)

In batch gradient descent, because every time you keep the whole optimized, so the loss function will from top to the bottom, but the stochastic gradient descent will just gradient descent randomly, extremely noisy, maybe to the bottom, maybe to another direction, and may not get the best solution just around the lowest cost.

![](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimageimage-20230919194450196.png)

In practice, the mini-batch size you use will be somewhere in between. The pure stochastic gradient descent may lose speed up from vectorization, and the pure batch gradient descent may cost much more time to per iteration. 

So we use the in-between mini-batch size, not too large and not too small, make neural network be training fastest and utilize vectorization for speed up.

![image-20230919194004983](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919194004983.png)!

The In-between mini-batch size is just as the green line shows, although it cannot down to the bottom finally, but will decrease the cost consistently.

If you find your neural network is around the optimized point, you can reduce the learning rate, this is **learning rate decay**.





## Exponentially weighted (moving) averages

Is there any optimization algorithms except gradient descent?

We have the temperature data about the London, we can plot the scatter figure, and then plot the curve using exponentially weighted averages.

With your $\beta$ increasing, your temperatures curve will more smooth, it focus on your previous data point but care less about current data value.

The exponentially weighted averages can smooth the data set, make it more friendly but not change highly.



![image-20230919200531456](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919200531456.png)

We need to find out **the essence of the exponentially weighted averages.**

You can expand the formulas, then you can get a huge like a chain formula, it about the $\theta_i$, and decay weights.

The final value can be compute by the multiple with weights and sample point value.

![S](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919201519895.png)

If your $\beta$ is 0.9, you may get value of 0.35 about exponential 10, so you just focus the latest 10 data set, if you use 0.98, you focus the latest 50 data, so the more $\beta$, the latest day you focus. We approximately average 1/(1-$\beta$) days minus.

How we can implement exponentially weighted averages?

Just repeat the update about the $V_\theta$, you just need store two value, the old $V_\theta$ and the current value, very efficiency and low cost.

![image-20230919211121205](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919211121205.png)



## Bias correction in exponentially weighted average

The average calculation above has little error, we want to correct the weighted average.

because of the first point of $v_0$, so the v1 and v2 is much less than the sample value of $\theta1$ and $\theta2$, so we need to do bias correction.

![image-20230919214505777](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230919214505777.png)

Because our early $V_t$ is incorrect, so you need to add some bias element to improve this. Our weight is decrease exponentially, so we can use exponential bias to modify the early data.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920064245174.png" alt="image-20230920064245174" style="zoom:67%;" />

People don't bother to implement bias corrections because most people would rather just wait that initial period and have a slightly more biased estimate and go from there. If you concern about the bias during this initial phase, bias correction can help you get a better estimate early on.





## Gradient descent with momentum

After we know about the exponentially weighted, we can learn the new gradient descent technique.

Analyze the original gradient descent first. Gradient descent will cost too much steps, just slowly oscillate toward the minimum, this oscillation prevent you to use a much larger learning rate or you might end up overshooting and end up diverging like so.

![image-20230920065544023](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920065544023.png)

You just want fast learning, from left to the bottom, but not from bottom to the top, reduce the oscillation of the training process.

![image-20230920084707838](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920084707838.png)

So we use momentum to speed the training step.

The momentum optimized process like our exponentially weight gradient, compute the average of gradient, for the vertical, the average value is almost 0 because of this oscillation, so we reduce the slower learning, and for the horizontal, the all derivative is direct to bottom, so the training step become much more fast.

The gradient descent is like the red line, more smooth, more quickly.

You can just treat the optimized procedure is a bowl rolling from top to the bottom. The bowl has its velocity and will get some acceleration during this rolling process.

![image-20230920091201707](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920091201707.png)

How to implement momentum gradient descent?

Because the iteration times is more than 10, so we not focus the initial value of $V$<sub>dw</sub> can just set 0,  

![image-20230920092210829](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920092210829.png)

There is another formula of the V<sub>dw</sub>, but if $\beta$ changes, you must modify the learning rate of $\alpha$ and dW and db, so we recommend to use left equation, just as exponentially weight average.

![image-20230920092629191](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920092629191.png)

Momentum is an improve to original optimized algorithm, but also a gradient descent algorithm, just smooth the gradient, make training step much more quickly and efficient.





## RMSprop

How to use another technique to optimized the neural network?

**Root means square propagation(RMSprop)**, squaring the derivatives and then you take the square root here at the end.

![image-20230920095007312](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920095007312.png)

We want to slow down all the oscillations into the vertical direction and want learning to go pretty fast.

The parameters update strategy is $dw/sqrt(S_w)$, if the S<sub>dw</sub> is a mall value, you need slow down the update speed, because the optimized is near to the optimized, but if the S<sub>dw</sub> is a huge value, the  S<sub>dw</sub> will decrease, become a small value. 

So this operation can keep the derivative value is always at the average level, not too huge, not too small.

The smaller value can cause big step, larger value may cause small step.

To just ensures slightly greater numerical stability, make sure your algorithm will not divide by 0, you need plus a $\epsilon$, just make it a small value but not 0.

![image-20230920112512527](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230920112512527.png)





## Adam optimization algorithm

The Adam optimization algorithm is basically taking momentum and RMSprop and putting them together.

This is a common algorithm, and is proven to be very effective for many different neural networks.

![image-20230921080910998](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921080910998.png)

In the Adam algorithm above, you have to chose property hyperparameters.

![image-20230921081222613](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921081222613.png)

Adam: **Adaptive moment estimation**. The dw is the first moment, and dw2 is the second moment, first is momentum, second is RMSprop.





## Learning rate decay

One of the things that might help speed up your learning algorithm is to slowly reduce your learning rate over time.

During your iterate, your steps will be a little bit noisy, so it won't exactly converge because you're using some fixed value for learning rate. The final step will still very large, so you cannot get the bottom.

If you reduce your learning rate during training step, you end up oscillating in a tighter region around this minimum.

In the early training step, you can afford big training steps, but in the approaches converges steps, you just need a small learning rate.

![image-20230921081727463](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921081727463.png)

Let's see how to implement it. We can use the formula to update $\alpha$, with the increasing of epoch, learning rate become smaller and smaller. The $\alpha_0$ and decay_rate is also the hyperparameter of this neural network.

![image-20230921082721612](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921082721612.png)

And there maybe another formula to update the learning rate.

You can use exponentially decay, it's very simple, or using discrete staircase, if t reach the number limit, then change to another learning rate.

![image-20230921083017858](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921083017858.png)

We will tune the hyperparameters rather than to try learning rate decay.





## The problem of local optima

saddle point (鞍点)

plateau (平稳段)

The intuition of low dimension cannot be used in high dimension, if you have 20000 parameters, you're much more likely to see saddle points but not the local optima.

![image-20230921094413982](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921094413982.png)

The local optima is not the problem, the truly problem is the plateaus, it's a region which the derivative is close to 0 for a long time. 

So you need a long time to slowly find your way to maybe this point on the plateau.

This plateau can use Adam 、RMSProp、Momentum to accelerate, although the dW and db is very small because of the derivative is close to 0, the Momentum will also keep the derivative bigger.

![image-20230921094855816](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230921094855816.png)
Just remember:

- Unlikely to get stuck in a bad local optima
- Plateaus can make learning slow



These is the challenges that the optimization algorithms may face, the algorithm is not always performance very well, but we can improve our technique to reduce the effect of plateaus.

