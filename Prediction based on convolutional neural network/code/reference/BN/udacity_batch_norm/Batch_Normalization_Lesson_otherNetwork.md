# Considerations for other network types

笔记本演示了具有完全连接层的标准神经网络中的批量标准化。您还可以在其他类型的网络中使用批量规范化，但有一些特殊的考虑因素

### ConvNets

卷积图层由多个特征图组成。(请记住，卷积层的深度指的是其特征图的数量)每个特征图的权重在所有输入图层中共享。因为这些差异，Batch Normalization操作需要在每个特征图上计算批数据的平均值分布和方差分布而不是以神经元层中的神经元作为单位。

使用tf.layers.batch_normalization时，一定要注意卷积维数的顺序。具体来说，一般图层的通道是作为卷积输入的最后一个维度，如果你自定义了图层的通道为第一个维度，一定要为axis参数设置不同的值。

在我们使用tf.nn.batch_normalization的低级实现时，我们使用以下语句计算批数据中的平均值和方差:
```python
batch_mean, batch_variance = tf.nn.moments(linear_output, [0])
```

如果我们使用相同的函数在卷积神经网络上使用Batch Normalization，我们需要使用以下语句替代计算批处理数据中的平均值和方差:
```python
batch_mean, batch_variance = tf.nn.moments(conv_layer, [0,1,2], keep_dims=False)
```                                              

第二个参数'[0,1,2]'告诉Tensorflow在每个特征图上计算批处理数据的平均值和方差。(这三个维度分别表示批处理数量,图片高度,图片宽度)设置'keep_dim'参数为'False'告诉'tf.nn.moments'函数不要返回和输入数据一样维度的计算结果，而是保证在每一个特征图上计算得到一个平均值和方差对。

### RNNs

Batch normalization 也可以运用到时序网络，在2016年的论文 [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025).这需要更多的工作来实现，但基本上涉及计算每个时间步的平均值和方差，而不是在每层上进行计算。你可以在[this GitHub repo](https://gist.github.com/spitis/27ab7d2a30bbaf5ef431b4a02194ac60)找到别人扩展的Batch Normalization在RNN上的应用'tf.nn.rnn_cell.RNNCell'函数。
