# 不使用tf.layers包提供的函数实现Batch Normalization
"""
我们在NeuralNet类中实现Batch Normalization使用的是Tensorflow中tf.layer包提供的高级函数tf.layers.batch_normalization
然而，如果你想使用低级函数实现Batch Normalization,你可以使用Tensorflow中的nn包中的低级函数tf.nn.batch_normalization 
你可以使用以下函数重写NeuralNet类中的fully_connected函数，NeuralNet类会和原来一样操作，实现一样的效果。
"""

import tensorflow as tf
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def fully_connected(self, layer_in, initial_weights, activation_fn=None):
    """
    Creates a standard, fully connected layer. Its number of inputs and outputs will be
    defined by the shape of `initial_weights`, and its starting weight values will be
    taken directly from that same parameter. If `self.use_batch_norm` is True, this
    layer will include batch normalization, otherwise it will not.
    创建一个标准的完全连接图层。 其输入和输出的数量将是由`initial_weights`的形状定义，其初始权重值将为
    直接从相同的参数中获取。 如果`self.use_batch_norm`为True，则为图层将包含Batch Normalization，否则不会。


    :param layer_in: Tensor
        The Tensor that feeds into this layer. It's either the input to the network or the output
        of a previous layer.
    :参数 layer_in: Tensor
    该层的输入张量，如果创建的是第一层，则此参数为整个网络的输入，如果为中间层，则为前一层的输出

    :param initial_weights: NumPy array or Tensor
        Initial values for this layer's weights. The shape defines the number of nodes in the layer.
        e.g. Passing in 3 matrix of shape (784, 256) would create a layer with 784 inputs and 256
        outputs.
    ：参数 initial_weights:numpy 数组或者是张量
    初始化该层的权重，其形状定义了该层的节点数量。
    例如：如果传入一个形状为(784,256)的矩阵，则会创建一个有784个输入，256个输出的神经层

    :param activation_fn: Callable or None (default None)
        The non-linearity used for the output of the layer. If None, this layer will not include
        batch normalization, regardless of the value of `self.use_batch_norm`.
        e.g. Pass tf.nn.relu to use ReLU activations on your hidden layers.
     ：参数 activation_fn:可调用或者没有(默认没有激活函数)
    用于输出神经层的非线性，如果该层没有使用激活函数，则无论self.use_batch_norm标志是否激活都不会使用Batch Normalization
    例如：使用tf.nn.relu函数来讲ReLU激活方法用于隐藏层
    """
    if self.use_batch_norm and activation_fn:
        # Batch normalization uses weights as usual, but does NOT add a bias term. This is because
        # its calculations include gamma and beta variables that make the bias term unnecessary.
        # Batch normalization 和平时一样使用权值，但是不用使用偏置项，这时我们需要额外计算gamma和beta这两个额外项而不用使用偏置属性
        weights = tf.Variable(initial_weights)
        linear_output = tf.matmul(layer_in, weights)

        num_out_nodes = initial_weights.shape[-1]  # 表示输出神经元的个数

        # Batch normalization adds additional trainable variables:
        # gamma (for scaling) and beta (for shifting).
        # Batch Normalization需要添加额外的可训练变量：
        # gamma(用于缩放)和beta(用于偏移)
        gamma = tf.Variable(tf.ones([num_out_nodes]))
        beta = tf.Variable(tf.zeros([num_out_nodes]))

        # These variables will store the mean and variance for this layer over the entire training set,
        # which we assume represents the general population distribution.
        # By setting `trainable=False`, we tell TensorFlow not to modify these variables during
        # back propagation. Instead, we will assign values to these variables ourselves.
        # 这些参数会存储这个神经层上整个训练集数据上的平均值和方差，假设它代表了训练数据的总体分布
        # 通过设置这些参数为trainable=False,我们告诉Tensorflow在反向传播时不要修改这些值，并且我们会传值给这些变量
        pop_mean = tf.Variable(tf.zeros([num_out_nodes]), trainable=False)
        pop_variance = tf.Variable(tf.ones([num_out_nodes]), trainable=False)

        # Batch normalization requires a small constant epsilon, used to ensure we don't divide by zero.
        # This is the default value TensorFlow uses.
        # Batch Normalization 需要一个很小的常数epsilon来确保分母不为0，1e-3是Tensorflow的默认值
        epsilon = 1e-3

        def batch_norm_training():
            # Calculate the mean and variance for the data coming out of this layer's linear-combination step.
            # The [0] defines an array of axes to calculate over.
            # 计算通过神经元线性输出(Wx)后的值得平均值和方差，其中[0]定义了对数组计算平均值和方差的维度
            batch_mean, batch_variance = tf.nn.moments(linear_output, [0])

            # Calculate a moving average of the training data's mean and variance while training.
            # These will be used during inference.
            # Decay should be some number less than 1. tf.layers.batch_normalization uses the parameter
            # "momentum" to accomplish this and defaults it to 0.99
            # 用计算滑动平均的方法对训练集上的数据平均值和方差进行预测，这在预测(只进行前向传播而不进行反向传播时)
            # decay是计算滑动平均值时使用的，其中decay=0.99表示统计1/(1-0.99)=100个数据的平均值，在tf.layers.batch_normalization
            # 函数中使用参数"momentum"表示decay,此值默认为0.99
            decay = 0.99
            # 手动调用滑动平均计算公式
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

            # The 'tf.control_dependencies' context tells TensorFlow it must calculate 'train_mean'
            # and 'train_variance' before it calculates the 'tf.nn.batch_normalization' layer.
            # This is necessary because the those two operations are not actually in the graph
            # connecting the linear_output and batch_normalization layers,
            # so TensorFlow would otherwise just skip them.
            # 'tf.control_dependencies'方法告诉程序必须计算在计算'tf.nn.batch_normalization'层时先计算train_mean和train_variance两个函数
            # 这两个函数实际上并不在图模型中所以使用tf.control_dependencies是必要的，通过这个操作连接线性输出和batch_normalization层，
            # 否则tensorflow会跳过他们
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)

        def batch_norm_inference():
            # During inference, use the our estimated population mean and variance to normalize the layer
            # 在前向传播推测的过程中使用估计的数据集平均值和方差来进行预测
            return tf.nn.batch_normalization(linear_output, pop_mean, pop_variance, beta, gamma, epsilon)

        # Use `tf.cond` as a sort of if-check. When self.is_training is True, TensorFlow will execute
        # the operation returned from `batch_norm_training`; otherwise it will execute the graph
        # operation returned from `batch_norm_inference`.
        # 使用`tf.cond`作为一种if - check.当self.is_training为True时,将执行TensorFlow从`batch_norm_training`返回的操作;
        # 否则它会执行图形从`batch_norm_inference`返回的操作
        batch_normalized_output = tf.cond(self.is_training, batch_norm_training, batch_norm_inference)

        # Pass the batch-normalized layer output through the activation function.
        # The literature states there may be cases where you want to perform the batch normalization *after*
        # the activation function, but it is difficult to find any uses of that in practice.
        # 将Batch Normalization输出后的值通过激活函数,BN算法的提出文章也提出过如果想要在激活函数后使用Batch Normalization，
        # 但是实际使用时没有遇见过这种情况
        return activation_fn(batch_normalized_output)
    else:
        # When not using batch normalization, create a standard layer that multiplies
        # the inputs and weights, adds a bias, and optionally passes the result
        # through an activation function.
        # 当不使用Batch Normalization时，创建标准多层感知机，如果不是最后一层使用激活函数输出(Wx+b)的值，如果是最后一层直接输出(Wx+b)的值
        weights = tf.Variable(initial_weights)
        biases = tf.Variable(tf.zeros([initial_weights.shape[-1]]))
        linear_output = tf.add(tf.matmul(layer_in, weights), biases)
        return linear_output if not activation_fn else activation_fn(linear_output)


"""
1）此版本的fully_connected比原始版本要长很多，但是再次充足的注释可以帮助您理解它。这里有一些重要的观点:
1. 它明确地创建变量来存储gamma，beta，以及总体均值和方差。这些都是在以前版本的功能中为我们处理的。
2. 将gamma初始化为1，β初始化为零，因此它们在计算的开始阶段不起作用。然而，在训练期间，网络使用反向传播学习这些变量的最佳值，就像网络通常对权重做的一样。
3. 与gamma和beta不同，计算总体均值和方差的变量被标记为不可排除。这告诉TensorFlow在反向传播期间不要修改它们。相反，调用tf.assign语句用于直接更新这些变量。 
4. TensorFlow在训练期间不会自动运行tf.assign操作，因为它只根据在图中找到的连接运行所需的操作。
   为了解决这个问题，我们添加tf.control_dependencies([train_mean，train_variance])在执行Batch Normalization之前，
   这使得Batch Normalization的更新操作必须在with语句块之内，先得执行平均值和方差变量的更新赋值才能进行Batch Normalization操作
5. 实际的标准化数学仍然大部分隐藏，这次使用tf.nn.batch_normalization。 
6. tf.nn.batch_normalization没有像tf.layers.batch_normalization那样的训练参数。但是，我们仍然需要以不同的方式处理训练和推理，
   因此我们使用tf.cond操作在每种情况下运行不同的代码。 
7. 我们使用tf.nn.moments函数来计算批处理均值和方差。


2）原版本NeuralNet类'train'函数可以适用于tf.nn.batch_normalization版本的fully_connected函数，
   然而，它会使用以下代码来确认当使用Batch Normalization时，数据分布是否发生变化:
   if self.use_batch_norm:
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    else:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    使用tf.nn.batch_normalization版本的fully_connected函数会直接更新数据分布，这表明你可以将tf.layer.batch_normalization版本中的以上代码用以下代码来简化:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


3) 如果你想从头开始实现Batch Normalization中的细节，你可以修改'batch_norm_training'中的这行代码:
   return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)
   以以下代码取代:
   normalized_linear_output = (linear_output - batch_mean) / tf.sqrt(batch_variance + epsilon)
   return gamma * normalized_linear_output + beta
   也可以将'batch_norm_inference'中的这行代码:
   return tf.nn.batch_normalization(linear_output, pop_mean, pop_variance, beta, gamma, epsilon)
   用以下代码进行取代:
   normalized_linear_output = (linear_output - pop_mean) / tf.sqrt(pop_variance + epsilon)
   return gamma * normalized_linear_output + beta
   
   你会发现这两句公式就是Batch Normalization算法的核心公式，第一句通过X的平均值和方差计算^Xi,第二句计算Yi=Lambda*^Xi+Beta
   其中linear_output表示Xi,normalization_linear_output表示^Xi
   
"""
