# Batch Normalization using tf.nn.batch_normalization
# 使用tf.nn.batch_normalization函数实现Batch Normalization操作

"""

大多数情况下，您将能够使用高级功能，但有时您可能想要在较低的级别工作。例如，如果您想要实现一个新特性—一些新的内容，那么TensorFlow还没有包括它的高级实现，
比如LSTM中的批处理规范化——那么您可能需要知道一些事情。

这个版本的网络的几乎所有函数都使用tf.nn包进行编写，并且使用tf.nn.batch_normalization函数进行标准化操作

'fully_connected'函数的实现比使用tf.layers包进行编写的要复杂得多。然而，如果你浏览了Batch_Normalization_Lesson笔记本，事情看起来应该很熟悉。
为了增加批量标准化，我们做了如下工作:
Added the is_training parameter to the function signature so we can pass that information to the batch normalization layer.
1.在函数声明中添加'is_training'参数，以确保可以向Batch Normalization层中传递信息
2.去除函数中bias偏置属性和激活函数
3.添加gamma, beta, pop_mean, and pop_variance等变量
4.使用tf.cond函数来解决训练和预测时的使用方法的差异
5.训练时，我们使用tf.nn.moments函数来计算批数据的均值和方差，然后在迭代过程中更新均值和方差的分布，并且使用tf.nn.batch_normalization做标准化
  注意：一定要使用with tf.control_dependencies...语句结构块来强迫Tensorflow先更新均值和方差的分布，再使用执行批标准化操作
6.在前向传播推导时(特指只进行预测，而不对训练参数进行更新时)，我们使用tf.nn.batch_normalization批标准化时其中的均值和方差分布来自于训练时我们计算的值
7.将标准化后的值通过RelU激活函数求得输出
8.不懂请参见https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Lesson.ipynb
  中关于使用tf.nn.batch_normalization实现'fully_connected'函数的操作
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)


def fully_connected(prev_layer, num_units, is_training):
    """
    Create a fully connectd layer with the given layer as input and the given number of neurons.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new fully connected layer
    """

    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)

    gamma = tf.Variable(tf.ones([num_units]))
    beta = tf.Variable(tf.zeros([num_units]))

    pop_mean = tf.Variable(tf.zeros([num_units]), trainable=False)
    pop_variance = tf.Variable(tf.ones([num_units]), trainable=False)

    epsilon = 1e-3

    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(layer, [0])

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)