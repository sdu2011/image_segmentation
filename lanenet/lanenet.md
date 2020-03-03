
# LaneNet
- LanNet　
    - Segmentation branch　完成语义分割,即判断出像素属于车道or背景
    - Embedding branch　完成像素的向量表示,用于后续聚类,以完成实例分割
- H-Net

## Segmentation branch
解决样本分布不均衡　　　

车道线像素远小于背景像素.loss函数的设计对不同像素赋给不同权重,降低背景权重.

该分支的输出为(w,h,2)．

## Embedding branch
loss的设计思路为使得属于同一条车道线的像素距离尽量小,属于不同车道线的像素距离尽可能大.即Discriminative loss.

该分支的输出为(w,h,n)．n为表示像素的向量的维度.

## 实例分割
在Segmentation branch完成语义分割,Embedding branch完成像素的向量表示后,做聚类,完成实例分割.

![](https://img2018.cnblogs.com/blog/583030/202003/583030-20200302112317801-1679623589.png)


## H-net
### 透视变换
to do

### 车道线拟合
LaneNet的输出是每条车道线的像素集合，还需要根据这些像素点回归出一条车道线。传统的做法是将图片投影到鸟瞰图中，然后使用二次或三次多项式进行拟合。在这种方法中，转换矩阵H只被计算一次，所有的图片使用的是相同的转换矩阵，这会导致坡度变化下的误差。
为了解决这个问题，论文训练了一个可以预测变换矩阵H的神经网络HNet，网络的输入是图片，输出是转置矩阵H。之前移植过Opencv逆透视变换矩阵的源码，里面转换矩阵需要8个参数，这儿只给了6个参数的自由度，一开始有些疑惑，后来仔细阅读paper，发现作者已经给出了解释，是为了对转换矩阵在水平方向上的变换进行约束。

## 测试结果
tensorflow-gpu 1.15.2
4张titan xp

(4, 256, 512) (4, 256, 512, 4)
I0302 17:04:31.276140 29376 test_lanenet.py:222] imgae inference cost time: 2.58794s

(32, 256, 512) (32, 256, 512, 4)
I0302 17:05:50.322593 29632 test_lanenet.py:222] imgae inference cost time: 4.31036s

类似于高吞吐量,高延迟.对单帧图片处理在1-2s,多幅图片同时处理,平均下来的处理速度在0.1s.

论文里的backbone为enet,在nvida 1080 ti上推理速度52fps.

对于这个问题的解释,作者的解释是
>  2.Origin paper use Enet as backbone net but I use vgg16 as backbone net so speed will not get as fast as that. 3.Gpu need a short time to warm up and you can adjust your batch size to test the speed again:)
一个是特征提取网络和论文里不一致,一个是gpu有一个短暂的warm up的时间.

我自己的测试结果是在extract image features耗时较多.换一个backbone可能会有改善.
```
   def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        print("***************,input_tensor shape:",input_tensor.shape)
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            t_start = time.time()
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )
            t_cost = time.time() - t_start
            glog.info('extract image features cost time: {:.5f}s'.format(t_cost))

            # second apply backend process
            t_start = time.time()
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse
            )
            t_cost = time.time() - t_start
            glog.info('backend process cost time: {:.5f}s'.format(t_cost))

            if not self._reuse:
                self._reuse = True

        return binary_seg_prediction, instance_seg_prediction

```


参考:https://www.cnblogs.com/xuanyuyt/p/11523192.html　　https://zhuanlan.zhihu.com/p/93572094