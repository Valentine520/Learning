# Setting up your goal

If you just want to improve your model, but you don't know how to improve, you may try every method, such as collect more data, use bigger or smaller neural network, try dropout, add L2 regularization and so on.

But you may spend lots of time to walking in a wrong way, so you need a method to check whether a method is right or not. 

So you need to know the strategies of analyzing machine learning problems.

![image-20231008111434127](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008111434127.png)



## Orthogonalization

One of the challenges with building machine learning systems is that there's so many things you could try, so many things you could change.

But the old scientists would know what to tune in order to deal with the specific problem or get the goal.

Just think about the TV, you have many knobs, maybe you can tune the lightness or the horizontal or vertical and so on. But if you change one knobs, the other will change slightly, you may never tune the image in the center of the TV.

So the designer must make these knobs orthogonalization, every knob has its own duty, the light or the sound. There isn't any relation between these knobs, so we can tune the TV separately, this is much easier to tune the TV.  The Car is also need to be orthogonalization.

This orthogonalization is also important in our neural network tuning, you must tune some parameters to achieve the specific goal. You know what you want and how to tune it.

![image-20231008113006311](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008113006311.png)

The knobs of Machine Learning:

- Fit training set well on cost function *(human intuition performance )*
- Fit dev set well on cost function
- Fit test set well on cost function
- Performs well in real world

So if your algorithm doing well in the training set, but not well in the dev set, this is over-fitting, you want to improve but not effect the training set performance.

Every problem need to use different method to improve, and we always not use early stopping, this can effect training and dev cost function performance, so is not very orthogonalization.

![image-20231008114534183](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008114534183.png)

But these method is orthogonalization absolutely? The Adam will not effect the dev performance?

I think this is a very complex problem, we can just try again and again to find what is the best method.



## Single number evaluation metric

 There are much more parameters will effect your model, but you need using single number to evaluate them, make your evaluation easily.

Just like the Precision and Recall, the Precision is your recognized cat what percentage is the true cat, and the Recall is in the whole true cats, how much can be recognized correctly.

You always care about both, you want if there is a true cat, you can correctly recognize it as cat, and you want to pull a large fraction of them as cats.

But how to use the single value to evaluate the combination, is Precision more important or the Recall more important? It's very difficult to judge your model.



![image-20231008231832722](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008231832722.png)

You need to find a new evaluation metric that combines precision and recall, the **F1 score**, calculate the harmonic mean of Precision and Recall. The F1 score is very important in training step, this can evaluate the model is better or not quickly, your iterating step may faster.

![image-20231008232518360](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231008232518360.png)

## Satisficing and optimizing metrics

Consider your classifier has Accuracy and Running time two parameters, you need calculate the F1 score to get the final score value.

You can just calculate cost use linear weight method, but it is very sedulous, the more property way is set a limitation of Running Time, because  you may just not care about the time very much, your model's accuracy is very important, so you just want the time is satisfying your limitation and optimizing the accuracy.

So the Accuracy is Optimizing metric and the Running time is Satisficing metric.

![image-20231009071914114](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009071914114.png)

If you have N metric, sometimes pick just one as optimizing metric is very useful, you want to do as well as is possible to optimize that, but for other metric, you just need them not very worse, attention, but not so strict.

But the estimation is based on the dataset, how to set the train\dev\test dataset is also very important.



## Train/dev/test distributions

If you training a cat classifier, you can chose Dev and Test set from different regions, but the different distribution will cause your model perform not very well.

You need keep your dev and test set in the same distribution, your team use Dev regions dataset to train the classifier but the test dataset may have huge difference between Dev sets.

Maybe you spend much more time to close the bull's eye, but you may be taught change the bull's eye location when you testing your model.

![image-20231009073946483](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009073946483.png)

Avoid the above situation, you need put your all datasets **randomly shuffle into dev/test sets**, **You need keep your dev and test set in the same distribution**.

![image-20231009074509592](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009074509592.png)

We just talk about the dev set and test set, the dev set create the dev metric to evaluation your neural network, and the test metric is to test the true data on your model.

They all used to keep your machine learning goal is really your need, set the direction you may close to the final target, and the training set is just how fast you may close to the target.

![image-20231009074618507](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009074618507.png)

## Size of dev and test sets

Old way of splitting data is just use fixed rating to split dataset.

![image-20231009165648276](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009165648276.png)

But when your dataset is huge enough, you don't need too much test set and dev set, you just need make your training set bigger.

Set your test set to be big enough to give high confidence in the overall performance of your system.

And some situation, you don't need test set to testing your model, you need just train and dev set. 

This concept is mentioned in the previous chapter.

Summarize, in the era of big data, the old rule of thumb of a 70/30 split, that no longer apply, the trend has been to use more data for training and less for dev and test.

**The dev set is to help you evaluate different ideas and pick this up, A or B better. And the test set is to test your final  cost bias.**

Just enough, don't need too much, but **training set need to be huge**.



## When to change dev/test sets and metrics

You are training a cat classifier, your estimation metric is the error of algorithm. But now, the Algorithm A has lowest error rate, but it will often send porn image to users, this is very intolerable, so you need make huge penalty if there exist porn image.

We can just change the metrics, make the error rate weight huge, if the mistake prediction is a porn image, we need to reduce the rating of porn, even though the cat recognition accuracy is very worse.

![image-20231009183920568](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231009183920568.png)



The key point is that if you're not satisfied with your old error metric, then don't keep coasting with an error metric you're unsatisfied with, instead try to define a new one that you think better captures your preferences in terms of what's actually a better algorithm.

We need define these two step separately, the setting of target is first step, and how to close to this target, such as gradient descent, is second step, these steps is separately.

![image-20231011092153761](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011092153761.png)

If you want recognize some special image such as low clarity, you need change your dev/test set to these low clarity to make your model fitting these change.

![image-20231011092837231](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011092837231.png)



## Why human-level performance?

Your training quality will increase quickly under the human-level, but if the machine performance is better than human, the performance will increase very slowly. Never get the best of model, this is bayes optimized error. 

Because at the limit of human or machine performance, the image will very blurry, and we don't know what is correct, so this part of image cannot be trained.

Fortunately, human-level performance can solve all most every problem in the real world, recognize a image or audio, and so on, so there isn't too much space for machine to improve.

![image-20231011093352676](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011093352676.png)

Machine learning need human to make (data, label) as training dataset, so machine just can do some duplication work for human, not some creative work.

![image-20231011094147362](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011094147362.png)



## Avoidable bias

Consider two situation, if your training and dev error is always 8 and 10, but different humans error can cause different improve strategy, if your humans error is 1, the proxy for bayes error is 1, **Human-level error as a proxy for Bayes error.** so your machine have huge space to improve, you need use larger training set, or more training step. 

But for 7.5 humans level, there isn't any improve space for machine, so you just need to improve your model's variance but not bias. This is called avoidable bias, because your training task's bias error is some value, you cannot get below.

So you need to know the human-level error, in the background to estimate your model, chose the correct method to improve, causes you in different scenarios to focus on different tactics.

![image-20231011095010338](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011095010338.png)



## Understanding human-level performance

The optimal error has got to be 0.5% or lower because the team of experienced doctors can achieve 0.5% error.

If you just want to make your trained machine useful, just like a typical doctor, you can treat human-level performance as 1% error, but if you want to stead of bayes-error you can use team of experienced doctors such as 0.5% error.

![image-20231011102542638](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011102542638.png)

The different human-level performance may cause you chose different strategy.

If human-level is typical doctor, the error maybe 1%, so we need to improve variance, and if the human-level is 0.5, we need to improve bias.

When machine performance close to human-level, the training step will become more difficult, because the little bit different between training error and dev error will make you confused about reduce variance or bias.

![image-20231011103303188](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231011103303188.png)

The difference between your estimate of Bayes error tells you how much avoidable bias there is , and the difference training error and dev error tells you how much variance is a problem, whether your algorithm's able to generalize from the training set to the dev set.

For simple, you can just think bayes error is 0, so you can calculate the avoidable bias and the variance to chose the correct method. In the before course we do this, but not very suitable, sometimes the data is very noisy.





## Surpassing human-level performance

Because we don't know the percent of avoidable bias and variance, which one we should to optimized, so the efficiency where you should make progress is very slow.

When your training performance is surpassing the human-level, it's hard to know what is the correct direction for human to optimized.

![image-20231012102627388](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231012102627388.png)

Some problem ML is significantly surpassing human-level performance, just like:

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231012102942513.png" alt="image-20231012102942513" style="zoom:80%;" />

These training structured data is very larger than human can learned, so machine can do better than human.





## Improving your model performance

The two fundamental assumptions of supervised learning.

- You can fit the training set pretty well. (Avoidable bias)
-  The training set performance generalized pretty well to the dev/test set.(Variance)

Compare human-level and training error, and compare training error and dev error, chose optimize bias or variance.

If you find bias or variance is a problem, you can change your NN architecture, such as RNN and CNN. 

![image-20231012104731616](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231012104731616.png)























