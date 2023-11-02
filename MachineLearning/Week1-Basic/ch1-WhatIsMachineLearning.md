# Machine Learning



## What is Machine Learning

#### Machine learning can do:

- Data Mining
- Applications can't program by hand 
- Self-customizing programs
- Understand human-learning



#### Defination

Machine Learning is Filed of study that gives computers the ability to learn without being explicitly programmed. 

In another word, machine learning is to design a computer program that can play with it self and the get more and more practical.

Machine Learning is also a program that learned from experience(E) with respect to some task(T) and some performance measure P.



#### Example:

a spam or not spam email filter.

1. Task
   An email is spam or not spam.
2. Experience
   The label of emails.
3. Performance measure
   The number of correct filter.





## Supervised Learning

just like this, we will tell machine a dataset which contain the right answer.

The goal of the algorithm is to give more right answer if there comes a new problem.

![image-20230818095033426](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818095033426.png)

This problem is **Regression problem,** predict continuous valued based on some discrete point.



Another example:

We get the tumor size and the malignant answer, figure out the relation between the tumor size and whether get malignant.

This is a **Classification** problem. The task is judge whether get malignant, and the E is tumor size data, and the P is the right classify cases. The discrete point in, the discrete result out.

![image-20230818095456630](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818095456630.png)

**The class probably more than 2, it can has 10 possibilities, 100 parameters and 10 output, the max possibility is the class the algorithm predict.**

Just like below, we have age and tumor size and more parameters,  the principal is same, just give the possibility about the task based on the given supervised data.

![image-20230818100236411](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818100236411.png)

But if we have infinite features, how to store in computer memory?

The SVM(Supported Vector Machine) can do this.



In summary, the supervised learning is given a dataset which **contain the correct answer**, and then we have a task which need to classify or get the right answer.



## Unsupervised Learning

We have the dataset is **a pure dataset**, there isn't any labels so we need to let the computer generate the label or achieve the target automatically.

The unsupervised learning can cluster the dataset automatically, **Cluster Algorithm**.

![image-20230818102819588](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818102819588.png)

Another example is the google news website, there are all kinds of million thousands news everyday, so **the company need to cluster the same group news,** let the user can get the information in the same group.

The bilibili website also do this, it can cluster the same group video so that you can visit them at the one page.

![image-20230818103203178](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818103203178.png)



And in the genes series, we need to cluster the genes series into some special type, but I don't know the type, it just depend on the computer.

![image-20230818103509633](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818103509633.png)

**We don't give the right answer to the program.  This is unsupervised learning**



we can use the cluster algorithm to analyze the people who are know each other.  

And do some market segmentation, divided the customer into n layers and customized the sell plan for the different layer people.

![image-20230818103746188](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818103746188.png)

We need the computer to find the useful information itself.

Just like this, the two microphone can catch the two human voice, so we need to split the overlapped voice. 

![image-20230818104922205](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230818104922205.png)



Octave / Matlab will make you learn fast. Use octave to build the algorithm and make sure it can work correctly, then migrate them into the C++ language and do more complicated applications.

