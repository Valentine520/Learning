# DataSet and DataLoader



The Dataset is the data, but do some pre-solution

Then the dataset transfer the data to the dataloader, dataloader define the data reading function, such as data batch size and so on.



![image-20230815080351725](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230815080351725.png)

The DataSet must keep the image and target is couppled

The image and target data need to be transfer into the DataSet init function



### DataLoader: How to get the data by batch

1. first, generate the index by the range of num of 

![image-20230815081708775](https://hacker-oss-typora.oss-cn-chengdu.aliyuncs.com/Learning/imageimage-20230815081708775.png)

