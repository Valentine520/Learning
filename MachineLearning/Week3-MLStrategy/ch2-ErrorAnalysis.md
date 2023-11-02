## Carrying out error analysis

If your classifier often recognize a dog as a cat, just like the follow two dogs, it look like cats. Should you to train a dog classifier to improve your model? you may spend a few months doing this, and maybe have no effort finally.

![image-20231013093635819](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013093635819.png)

There are some tips may help you to find the correct direction. This is **error analysis**.

- Get ~100 mislabeled dev set examples.
- Count up how many are dogs.

If there just 5 dog error in the 100 error examples, the few months cost is not very worthy, because your error might go down from 10% error, down to 9.5% error.

This analysis gives you a ceiling, upper bound on how much you could improve performance by working on the dog problem.

your statistic maybe just cost 5~10 minutes, to avoid a few months non-effort training.

### Evaluate multiple ideas in parallel

Some ideas like dogs being recognized as cats, also can be evaluated in single error analysis, just like:

- Fix pictures of dogs being recognized as cats
- Fix great cats(lions, panthers, etc..) being misrecognized
- Improve performance on blurry images

You can use a table to do this statistic, the column is your analysis aim, you can analysis many ideas in single table, counts the percentage of these error, to check which one is the most dangerous.

If the error rate of Blurry is 61%, very high, you need first care about the image blurry problem.

These analysis are in your dev set datasets, help you to find which one is the most important to improve, and maybe you can get some new ideas for your model.

![image-20231013095127906](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013095127906.png)



## Cleaning up Incorrectly labeled data

If you find there are incorrect label in your dataset, just like the dog being recognized cat, but fortunately, DL algorithms are quite robust to random errors in the training set.

But DL algorithms cannot deal with the systematic errors, like the researches label all the cat images as dogs.

![image-20231013100247307](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013100247307.png)

If you care about the incorrectly labeled data in dev/test dataset, you can just add new column in the error analysis table, count the impact on a 100 mislabeled dev set examples, the original dev set incorrect can also treat as the normal ideas, the error analysis can tell you whether the mislabel dangerous or not.

You can see the error percentage is just 6%, not very worse, so you may not fix it. You can just simple calculate, your overall dev set error is 10%, so the errors due incorrect labels is just 0.6%, is very small, you can just ignore. But the errors due to other causes is 9.4%, very huge, you may first solve this problem.

![image-20231013101106416](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013101106416.png)

If you find the incorrect labels has huge effective for your model, you may check your dev/test data, and correct incorrect dev/test set examples, and there also some principles:

- Apply same process to your dev and test sets to make sure they continue come from the same distribution.
- Consider examining examples your algorithm got right as well as ones it got wrong. (It's hard to do)
- Train and dev/test data may now come from slightly different distributions. (Just make sure your dev/test set come from same distribution, the train set is very robust)

The deep learning process is not pure machine learning, you can just feed your data set to your model, and train it, but the result is not very well. 

Building practical systems, often there's also **more manual error analysis and more human insight that goes into the systems**.

Some researches be reluctant to manually look at the examples, but these is very useful, can really help to figure out which problem may solve firstly.



## Build your first system quickly, then iterate

**Speech recognition example**:

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013105916596.png" alt="image-20231013105916596" style="zoom:80%;" />

For deep learning progress, there maybe lots of direction to think about these problem. Just chose one direction quickly and build your first simple system, and then to optimize it, first set up dev/test set and metric. Use Bias/Variance analysis and Error analysis to prioritize next steps. 

Let you know your initial system drawback, and you can improve your mode. **Just build your system quickly, then iterate.** Don't overthink.

![image-20231013112313259](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231013112313259.png)

## Training and testing on different distributions

On the big data era, people often collect training data from everywhere to make its strong enough to train the model, but often come from different distribution with testing data.

Just remember, your dev/test set is to set the target of your nn model, like there are two type of data from different way, your application is to recognize the image which users upload from mobile app, so you need to set the target as the data from mobile app.

The option 1 shuffle the dataset, is not very well, because the dev\test set has only few mobile app data, the target is not explicitly.

The option 2, make sure the dev/test set is fill with the mobile app data, make the target right.

![image-20231015085833374](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231015085833374.png)



Consider another situation, the speech recognition example:

Your training set is complex and huge enough, because it contains all kinds of data, but not always the speech activated rearview mirror, so you just need set the dev/test set the speech activated rearview mirror data, but the training set must contain part of them.

![image-20231015090558410](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231015090558410.png)



Allow the training set and dev/test set from different distribution make your training set huge enough, and is all these dataset which you collect must be used?



## Bias and Variance with mismatched data distributions

Consider a cat classifier example. If your training set and dev set come from different distribution, maybe the training set is clear image but the dev set is blurry image. So we improve bias or variance? It's hard to answer, because the difference of error is just because the different dataset, maybe the blurry set has more error rating.

![image-20231016103346588](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016103346588.png)

In order to check whether optimized the bias or variance, you can make a new dataset that is training-dev set, the same distribution as training set, but not used for training, just for verifying.

The NN model save the training set, but if the model cannot do well at the same distribution of training set which called training-dev set, you may have variance problem in your NN model.

![image-20231016103717848](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016103717848.png)

If the training-dev error is just 1.5%, close to the training error, it stand for your NN model doesn't have any variance problem, the  error between dev set and training set is just because the original dataset difference.

This is the data mismatch problem, whatever algorithm it's learning, it works great on training-dev set but it doesn't work well on dev. Your algorithm has learned to do well on a different distribution.

![image-20231016103955263](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016103955263.png)

If your training error is 10% and dev error is 20%, there are two issues, the bias and the data mismatch problem or variance problem.

You can use 5 error to figure out your NN model's problem, the avoidable bias 、variance、data mismatch 、degree of over fitting to dev set.

If your model performance well on the dev set but not well on the test set, maybe there exists the dev set over tuning issue.

And you can see the right error data, **the dev error is lower than training set error, because of the data different distribution, this is also the data mismatch problem**.

![image-20231016105344422](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016105344422.png)

**More general formulation**

Just check the human level difference and training error and training-dev error and dev/test error to find your optimized direction. 

![image-20231016110518476](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016110518476.png)

Now we have the new issue, the data mismatch problem, so how to deal with this problem?

Unfortunately, there aren't super systematic way to address data mismatch.



## Addressing data mismatch

there aren't super systematic way to address data mismatch, but just something you could try.

**First, Carry out manual error analysis to try to understand difference between training and dev/test sets.**

You must check your training sets and dev/test sets to see what is the difference between training sets and dev sets actually. If your training data is the car rear-review audio, you may listen them carefully, **figure out the actual difference between training sets and dev/test sets**.



**Second, Make training data more similar; or collect more data similar to dev/test sets.**

If there isn't enough data of your training data set, you can just simulate noisy in-car data, to create some data manually, this is called artificial data synthesis.

Maybe you just have few in-car audio, but you have a large data set about the car noise, so you can use the sound to combine with the car noise to create a new in-car audio data.

![image-20231016171059019](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016171059019.png)

There is a potential risk of the artificial data synthesis, if you just have 1 hours car noise data, but you may have 1000 hours human audio data, your model maybe overfitting to the 1 hours car noise data.

You need collect 1000 hours of car noise, so you don't need to repeat the 1 hours noise again and again, solving the overfitting problem.

For human, the 1000 hours of car noise all sound the same as this one hour, so this is also a problem.

Consider another example, the car recognition. You can use the computer synthesis car image to train your NN model, it looks like no problem, even have good effect. But the difference cannot be saw in you eyes, this can also cause **the little subset overfitting**. 

If your training set is just from the video games, there maybe just 20 kinds of car in the games, but in the real world, there maybe lots of cars, so if you use the video games to train your model, you may encounter the little subset overfitting problem, and it's difficult for a person to easily tell that.

So you just need to prevent this little subset overfitting problem.

![image-20231016171857290](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016171857290.png)

## Transfer learning

Transfer learning is taking the knowledge the neural network has learned from one task and applying that knowledge to a separate task, this is one of the most important concept in the deep learning.

If you already trained a image recognition NN model, but now you just want to train a radiology diagnosis NN model, what you can do is change the last layer of the image recognition, don't change the previous layer's weight and structure.

You need the transfer learning just because you just have few data, cannot train a big NN model to achieve your goal, but you have huge enough data set to train a similar NN model such as the image recognition, so you can transfer the learning from image recognition to radiology diagnosis, just change the final layer.

The final layer can be trained by the radiology diagnosis data sets, so you can get a complete NN model to accomplish the radiology diagnosis.

Sometimes, not always the last layer, you can use a multiple new layer to replace the previous single layer, if you have enough data, in order to accomplish more complex issues.

The previous NN model is called **pre-training**, the little bit change is called **fine-tuning**.

![image-20231016173517250](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016173517250.png)

If you have 50 million data set, you can just train a new NN model, because your data set is huge enough, if you still use transfer learning, your pre-training model may just trained by 20 million data sets, this transfer learning may not have meaning grain.

**When transfer learning makes sense.**

- Task A and B have the same input x.
- You have a lot more data for Task A than Task B.
- Low level features from A could be helpful for learning B.

The first one keep the NN model can be very similar, the second and last one can keep the transfer learning is useful. Your pre-training NN model may have much more feature which is the basic of task B, like the radiology diagnosis, must base on the image recognition.

 Sometimes, you may difficult to get that many x-ray scans to build a good radiology diagnosis system, in that case, you might find a related but different task, where you have trained using millions of data, and learn a lot of low level features from that.



## Multi-task learning 

In transfer learning, you have a sequential process, where you learn from task A, and transfer to the task B. 

**But in the multi-task learning,** **you start off simultaneously**, **trying to have one neural network do several things at the same time, and then each of these task helps hopefully all of the other task**.

![image-20231016191101099](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016191101099.png)

If you want to build a image recognition system, to recognize the pedestrians or traffic lights or cars or stop signs and so on. You can just stack your goal into a vector, as the y_hat, just like [0, 0, 1, 1] is stand for there are stop signs and traffic lights in our image.

So the NN model's last layer has four nodes, this is different between the softmax regression. One image can have multiple labels, and train them simultaneously, this is multi-task training.

You can also training four separate NN model each one achieve a single goal, but in your NN model, the early layer, like layer 1, 2 ,3, may just as the same, because this layer just deal with normal image, extract the normal features of the image, the only difference between these task maybe the late layers, like transfer learning.

So the multi-task learning has more efficiency than the single task learning.

If your label matrix have some weird symbol, just like the ?, not 1 or 0, you can just ignore when you calculate the cost of this NN model.



![image-20231016191040846](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016191040846.png)

**When multi-task learning makes sense**.

- Training on a set of tasks that could benefit from having shared lower-level features.
- Usually: Amount of data you have for each task is quite similar. (not always useful)
  (If you already have 1,000 examples for single task, for all of the other tasks you better have a lot more than 1,000 examples, only this can help you do better on this final task.)
- Can train a big enough neural network to do well on all the tasks. (if your nn model is very small, the multi-task learning may make the performance be worse)

multi-task learning enables you to train one neural network to do many tasks and this can give you better performance than if you were to do the tasks in isolation.

multi-task learning is used much less often than transfer learning, because the complex of multi-task learning, and if your data set is very small, you can really improve your NN model through transfer learning. In the image recognition or object vision, people often training a huge NN model to detect lots of different objects.

They all helpful tools in your deep learning process.



## End-to-end deep learning

what is end-to-end learning?

Consider a speech recognition example, you may have an audio, and the old method is a long procedure to get the final transcript. First, use MFCC and get the features you want to know, and use ML to get the Phonemes, and combine into words and final get the transcript.

The end-to-end recognition is just from input to output, input is your audio, and out put is the final transcript.

![image-20231016224828558](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016224828558.png)

With the developing of end-to-end learning, the original pipeline designer will lose their jobs.

Consider another example, the face recognition, you can feed your NN model with the pure image, and make them to identity who you are. It's very hard, because there maybe many human need to be identity.

But if we split the whole procedure into two single isolate step, first find where is the face, and zoom 、cut and so on, then feed the image which be processed to the NN model.

<img src="https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016225547412.png" alt="image-20231016225547412" style="zoom:80%;" />

There are lot of data for the 2 subtask, more than the whole single task. You don't have enough data to solve this end-to-end problem, but you do have enough data to solve the 2 sub-problems, in practice, **split the whole task can have better performance than a pure end-to-end deep learning approach.**

So, don't use end-to-end learning blindly, you must consider whether better or not.



### More examples

Machine translation:

The translation can use end-to-end method very well, because you may have lots of (English, French) data pair to train your model, you input English text, and maybe find the pair, so you can get the French very quickly.

![image-20231016230923512](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016230923512.png)

Estimating child's age:

The old method is split the bones from image, and check the table to judge the age of this child. But the end-to-end method is not very well, the (Image, age) pair is not huge enough to train the model.

You can just split into two steps, two simple steps.

![image-20231016231233276](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20231016231233276.png)

Just remember,  **the end-to-end method is not panacea**.



# Whether to use end-to-end learning



## Pros and cons of end-to-end deep learning

**Pros:**

**First, the end-to-end learning may let the data speak.**

Your neural network learning input from x to y may be more able to capture whatever statistic are in the data, rather than being forced to reflect human preconceptions.

In the audio recognition, early has a phoneme concept, this is human created, not the data original exist.

**Second, Less hand-designing of components needed.**

This could also simplify your design work flow.



Cons:

**First, may need large amount of data.**

Like the face recognition machine, you may have huge data for face location and face recognition, but combine them you may have few data, which cannot support you to train your model through the end-to-end approach.

**Second, Excludes potentially useful hand-designed components.**

Just like the child age recognition, maybe the bones length is the best way to recognize, the bones is a useful hand-designed components.



Applying end-to-end deep learning, the key question is:

**Do you have sufficient data to learn a function of the complexity needed to map x to y?**



































