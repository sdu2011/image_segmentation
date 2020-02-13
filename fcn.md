# 全卷积网络FCN
fcn是深度学习用于图像分割的鼻祖.后续的很多网络结构都是在此基础上演进而来.


语义分割的基本框架:
前端fcn(以及在此基础上的segnet,deconvnet,deeplab等) + 后端crf/mrf


FCN是分割网络的鼻祖,后面的很多网络都是在此基础上提出的.
![论文传送门](https://arxiv.org/abs/1411.4038)


## 反卷积(deconvolutional)
关于反卷积(也叫转置卷积)的详细推导,可以参考:<https://blog.csdn.net/LoseInVain/article/details/81098502＞

简单滴说就是:卷积的反向操作．以4x4矩阵Ａ为例,卷积核Ｃ(3x3,stride=1),通过卷积操作得到一个2x2的矩阵B.

![](https://img2018.cnblogs.com/blog/583030/202002/583030-20200211145933264-1331554035.png)
和传统的分类网络相比,

## 代码解析
源码:<https://github.com/pochih/FCN-pytorch>
