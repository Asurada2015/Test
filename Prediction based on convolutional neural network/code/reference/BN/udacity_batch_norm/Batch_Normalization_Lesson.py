# 什么是Batch Normalization
"""
Sergey Ioffe和Christian Szegedy在2015年发表的论文“Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”
https://arxiv.org/pdf/1502.03167.pdf
中引入了Batch normalization。这个想法是，我们不仅对网络输入进行规范化，而是将网络内各层的输入标准化。
因为在训练过程中，我们使用mini-batch中的值的均值和方差对每个图层的输入进行归一化。这就是所谓的“batch”归一化

为什么这可能有帮助？
我们可以这样想，我们知道规范输入到网络有助于网络的学习。但是网络是一系列的层，其中一层的输出成为另一层的输入。这意味着我们可以将神经网络中的任何层看作小网络的第一层。

例如，想象一个3层网络。不要将其视为具有输入，隐藏层和输出层的单一网络，而第一层的输出看作是第二层和第三层组成的网络的输入。这个两层网络将由我们原始网络中的第2层和第3层组成。

类似地，第二层的输出可以被认为是仅由第三层组成的单层网络的输入

当你把它想成这样 - 作为一系列相互传值的神经网络 - 那么很容易想象如何规范化输入到每一层会有所帮助。这就像对任何其他神经网络的输入进行归一化处理一样，但是你要在每一层（子网络）上进行。

除了直观的原因之外，也有很好的数学原因解释为什么BN算法可以帮助网络更好地学习。它有助于对抗作者所说的"内部协变量转换"。
具体原因在论文[https://arxiv.org/pdf/1502.03167.pdf]和作者所著的深度学习[http://www.deeplearningbook.org/]中学习，
您可以在线阅读Ian Goodfellow，Yoshua Bengio和Aaron Courville在线阅读的书。
具体而言，请查看第8章：深度模型培训优化的批处理标准化部分[http://www.deeplearningbook.org/contents/optimization.html]。
"""

# Batch Normalization的优点
"""
batch normalization可优化网络训练。它已被证明有以下几个好处:

1.网络训练会更快
因为每次正向传播时会进行额外的计算，并且需要额外的超参数用于计算反向传播，所以训练过程会逐渐变慢。但是通过Batch Normalization
网络可以更快的收敛，所以总体来说会更快。

2.允许更高的学习率
梯度下降算法为了使网络收敛往往需要小的学习率。随着网络的深度加深，梯度变小，模型需要更多的迭代才能收敛。
使用Batch Normalization,可以允许我们使用更大的学习率，从而进一步提高网络收敛的速度。

3.使权重更加容易初始化
权重初始化可能很困难，尤其是创建更深的网络时更加困难，Batch Normalization 让我们在选择起始权重时不再那么谨慎。

4.使更多的激活函数变得可行
很多非线性函数在某些情况下表现不佳。例如Sigmoid很快会失去梯度，这以为这它们不能用于训练深度神经网络。
而且在训练时，Relu函数也有时会完全停止学习，在训练时死去、所以我们需要小心注入的数值范围。因为Batch Normalization 调节进入每个激活函数的值
所以在深度神经网络中似乎不起作用的非线性函数再次实际上变得可行。

5.简化更深层网络的创建
由于前面列出的四个理由，使用BatchNormaization时，创建和加速训练更深层神经网络会变得便捷，而且有研究表明使用更深层的神经网络会取得更好的结果。

6.提供一些正则化
Batch Normalization 向你的网络中添加了一些噪声，例如在Inception模块中，Batch Normalization 已经显示出和Dropout一样的效果。但总体来说，
因为其向网络中添加了额外的正则项，这允许你可以减少添加到网络中的某些丢失。

7.总体来说会得到更好的结果
一些测试似乎显示批量标准化实际上改进了培训结果。然而，这对于帮助更快速的训练是一种真正的优化，所以您不应该将其视为让网络更好的一种方式。
但是，由于它可以让你更快地训练网络，这意味着你可以更快速地迭代更多设计。它还可以让你建立更深的网络，这通常会有更好的效果。
所以，当你考虑所有事情时，如果你使用批量规范化建立你的网络，你可能会得到更好的结果。
"""

# Batch Normalization in TensorFlow
"""
notebook的这一部分向你展示了一种将批量标准化添加到TensorFlow内置的神经网络中的方法。 
以下单元格将笔记本中所需的包导入并加载MNIST数据集以用于我们的实验。但是，tensorflow软件包包含Batch Normalization实际需要的所有代码。
"""
# Import necessary packages
import tensorflow as tf
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data so we have something for our experiments
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Neural network classes for testing
"""
以下类NeuralNet允许我们创建具有和不具有Batch Normalization的相同神经网络。代码包含有大量的w文本，但稍后还有一些额外的讨论。
在阅读Notebook的其余部分之前，您不需要全部阅读，但代码块中的注释可能会回答您的一些问题。

关于代码：
这个类并不意味这是TensorFlow最佳实践 - 这里所做的设计选择是为了支持与批量标准化相关的讨论。 
同样重要的是要注意，我们在这些示例中使用了众所周知的MNIST数据，但我们创建的网络并不意味着对执行手写字符识别有好处。
我们选择了这种网络架构，因为它与原始论文中使用的相似，这足够复杂，足以证明Batch Normalization的一些好处，同时仍然快速训练。
"""


class NeuralNet:
    def __init__(self, initial_weights, activation_fn, use_batch_norm):
        """
        Initializes this object, creating a TensorFlow graph using the given parameters.
        初始化这个对象，使用给出的参数构建一个Tensorflow的graph模型
        :param initial_weights: list of NumPy arrays or Tensors
            Initial values for the weights for every layer in the network. We pass these in
            so we can create multiple networks with the same starting weights to eliminate
            training differences caused by random initialization differences.
            The number of items in the list defines the number of layers in the network,
            and the shapes of the items in the list define the number of nodes in each layer.
            e.g. Passing in 3 matrices of shape (784, 256), (256, 100), and (100, 10) would
            create a network with 784 inputs going into a hidden layer with 256 nodes,
            followed by a hidden layer with 100 nodes, followed by an output layer with 10 nodes.

        参数：initial_weights: Numpy数组或张量的列表
            为神经网络中每一层的权重赋值，我们将这些传入，以便我们可以创建具有相同起始权重的多个网络，以消除随机初始化差异导致的训练差异。
            列表中项目的数量定义了网络中的图层数量，列表中项目的形状定义了每个层中的节点数量。
            例如传递形状为(784,256),(256,100)和(100,10)的3个矩阵将创建一个具有784个输入的网络，进入具有256个节点的隐藏层，
            随后是具有100个节点的隐藏层，随后是10个节点的输出层。

        :param activation_fn: Callable
            The function used for the output of each hidden layer. The network will use the same
            activation function on every hidden layer and no activate function on the output layer.
            e.g. Pass tf.nn.relu to use ReLU activations on your hidden layers.

        参数： 激活函数: 可调用
            用于输出每个隐藏层的函数。网络将在每个隐藏层上使用相同的激活功能，并且在输出层上不使用激活功能。
            例如通过tf.nn.relu在隐藏层上使用ReLU激活函数。

        :param use_batch_norm: bool
            Pass True to create a network that uses batch normalization; False otherwise
            Note: this network will not use batch normalization on layers that do not have an
            activation function.
        参数：use_batch_norm: bool
            如果传入Bool值为True,则会创建一个使用Batch Normalization的神经网络，如果传入值为False,则会创建一个不使用Batch Normalization的神经网路
            注意：不会在没有激活函数的层上使用Batch Normalization
        """
        # Keep track of whether or not this network uses batch normalization.
        # 跟踪与标志这个网络是否使用Batch Normalization
        self.use_batch_norm = use_batch_norm
        self.name = "With Batch Norm" if use_batch_norm else "Without Batch Norm"

        # Batch normalization needs to do different calculations during training and inference,
        # so we use this placeholder to tell the graph which behavior to use.
        # Batch normalization在训练和前向传播时会使用不同的计算
        # 所以我们会用placeholder向图模型中传递我们会使用哪种行为
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # This list is just for keeping track of data we want to plot later.
        # It doesn't actually have anything to do with neural nets or batch normalization.
        # 这个list只是为了保存我们将要画出的图形所需要的数据
        # 对于神经网络和BatchNormalization算法没有额外的作用
        self.training_accuracies = []

        # Create the network graph, but it will not actually have any real values until after you
        # call train or test
        # 创建神经网络图模型，但是在你调用train或test之前，其不会有实值
        self.build_network(initial_weights, activation_fn)

    def build_network(self, initial_weights, activation_fn):
        """
        Build the graph. The graph still needs to be trained via the `train` method.
        构建图模型，这个图模型仍然需要使用train函数来运行训练操作
        :param initial_weights: list of NumPy arrays or Tensors
            See __init__ for description.
        :param activation_fn: Callable
            See __init__ for description.
        参数：initial_weights:numpy数据数组或者张量构成的列表
        参数：激活函数：可调用
        """
        self.input_layer = tf.placeholder(tf.float32, [None, initial_weights[0].shape[0]])
        # 如果这里形状为[(784,256),(256,100),(100,10)]的列表，则initial_weights[0].shape=(784,256)
        layer_in = self.input_layer  # 784 表示第一层的神经元数量
        for weights in initial_weights[:-1]:
            layer_in = self.fully_connected(layer_in, weights, activation_fn)

        # 由于这个函数是在for循环语句中创建的，并且需要用到initial_weights[:-1]这个参数
        # 所以我们必须知道initial_weight的定义以及fully_connected函数的定义以及返回值，
        # 程序中看出本层fully_connected函数的返回值会传入下一层的函数作为参数
        # 最后一层单独定义initial_weights[-1]表示输出层神经元个数
        self.output_layer = self.fully_connected(layer_in, initial_weights[-1])

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
        # Since this class supports both options, only use batch normalization when
        # requested. However, do not use it on the final layer, which we identify
        # by its lack of an activation function.
        # 我们只会在use_batch_norm标志被激活时使用BN算法，但是无论标志是否激活,在最后一层都不会使用BN算法,因为最后一层没有添加非线性激活函数

        if self.use_batch_norm and activation_fn:
            # Batch normalization uses weights as usual, but does NOT add a bias term. This is because
            # its calculations include gamma and beta variables that make the bias term unnecessary.
            # (See later in the notebook for more details.)
            # Batch normalization 和平时一样使用权值，但是不用使用偏置项，这是我们需要额外计算gamma和beta这两个额外项而不用使用偏置属性
            weights = tf.Variable(initial_weights)
            linear_output = tf.matmul(layer_in, weights)  # 线性输出

            # Apply batch normalization to the linear combination of the inputs and weights
            # 在神经层的输入和权值的线性组合上使用Batch Normalization
            batch_normalized_output = tf.layers.batch_normalization(linear_output, training=self.is_training)

            # Now apply the activation function, *after* the normalization.
            # 在使用BN算法之后使用非线性激活函数
            return activation_fn(batch_normalized_output)
        else:
            # When not using batch normalization, create a standard layer that multiplies
            # the inputs and weights, adds a bias, and optionally passes the result
            # through an activation function.
            # 当不使用Batch Normalization时，创建一个使用权值和输入相乘后加上偏置的标准层然后徐选择性的添加激活函数
            weights = tf.Variable(initial_weights)
            biases = tf.Variable(tf.zeros([initial_weights.shape[-1]]))  # 表示该层神经元的输出个数
            linear_output = tf.add(tf.matmul(layer_in, weights), biases)
            # 如果没有激活函数直接返回该值，有激活函数则通过激活函数计算后返回该值
            return linear_output if not activation_fn else activation_fn(linear_output)

    def train(self, session, learning_rate, training_batches, batches_per_sample, save_model_as=None):
        """
        Trains the model on the MNIST training dataset.
        在MNIST训练数据集上训练模型
        :param session: Session
            Used to run training graph operations.
        :参数 Session: Session
            用于运行训练图操作
        :param learning_rate: float
            Learning rate used during gradient descent.
        :参数 学习率：float
            梯度下降中使用的学习率
        :param training_batches: int
            Number of batches to train.
        :参数 训练批次数：int
            训练的批次数
        :param batches_per_sample: int
            How many batches to train before sampling the validation accuracy.
        :参数 batches_per_sample：int
            在抽样验证准确度之前要训练多少批次。
        :param save_model_as: string or None (default None)
            Name to use if you want to save the trained model.
        : 参数 save_model_as:string or None (default None)
            如果您想保存训练好的模型，请使用该名称
        """
        # This placeholder will store the target labels for each mini batch
        # 该占位符将存储每个小批量的目标标签
        labels = tf.placeholder(tf.float32, [None, 10])

        # Define loss and optimizer
        # 定义loss和优化器
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.output_layer))

        # Define operations for testing
        # 定义计算准确率的方法
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if self.use_batch_norm:
            # If we don't include the update ops as dependencies on the train step, the
            # tf.layers.batch_normalization layers won't update their population statistics,
            # which will cause the model to fail at inference time
            # 如果我们不包含更新操作作为训练操作的依赖关系，tf.layers.batch_normalization层不会更新均值和方差的统计值
            # 这会导致模型在前向传播的过程中失败，在训练时也要更新参数数值
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        else:
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        # Train for the appropriate number of batches. (tqdm is only for a nice timing display)
        # 训练合适的批次数量（tqdm只在最佳的时间显示，tqdm是一个显示条模块）
        for i in tqdm.tqdm(range(training_batches)):
            # We use batches of 60 just because the original paper did. You can use any size batch you like.
            # 我们使用60批次，仅仅是因为原论文这样描述,你可以将其调节为任意批次大小
            batch_xs, batch_ys = mnist.train.next_batch(60)
            # 在train函数调用时将Sess也传入，这样可以在train函数直接使用当前传入的session
            session.run(train_step, feed_dict={self.input_layer: batch_xs,
                                               labels: batch_ys,
                                               self.is_training: True})

            # Periodically test accuracy against the 5k validation images and store it for plotting later.
            # 在5k验证图片集上计算测试准确率，并且将其保存下来用于画图
            if i%batches_per_sample == 0:
                test_accuracy = session.run(accuracy, feed_dict={self.input_layer: mnist.validation.images,
                                                                 labels: mnist.validation.labels,
                                                                 self.is_training: False})
                self.training_accuracies.append(test_accuracy)

        # After training, report accuracy against test data
        test_accuracy = session.run(accuracy, feed_dict={self.input_layer: mnist.validation.images,
                                                         labels: mnist.validation.labels,
                                                         self.is_training: False})
        print('{}: After training, final accuracy on validation set = {}'.format(self.name, test_accuracy))

        # If you want to use this model later for inference instead of having to retrain it,
        # just construct it with the same parameters and then pass this file to the 'test' function
        # 如果你需要复用这个模型来预测，仅仅是前向传播不对其参数进行再次训练
        # 仅仅是使用同样的参数来重新构建这个模型，并且在这个模型上使用test函数
        if save_model_as:
            tf.train.Saver().save(session, save_model_as)
        # 如果save_model_as参数有值得话就保存模型

    def test(self, session, test_training_accuracy=False, include_individual_predictions=False, restore_from=None):
        """
        Trains a trained model on the MNIST testing dataset.
        在MNIST测试集上训练一个已经训练好的模型
        :param session: Session
            Used to run the testing graph operations.
        :param test_training_accuracy: bool (default False)
            If True, perform inference with batch normalization using batch mean and variance;
            if False, perform inference with batch normalization using estimated population mean and variance.
            Note: in real life, *always* perform inference using the population mean and variance.
                  This parameter exists just to support demonstrating what happens if you don't.
        :param include_individual_predictions: bool (default True)
            This function always performs an accuracy test against the entire test set. But if this parameter
            is True, it performs an extra test, doing 200 predictions one at a time, and displays the results
            and accuracy.
        :param restore_from: string or None (default None)
            Name of a saved model if you want to test with previously saved weights.
        """
        # This placeholder will store the true labels for each mini batch
        labels = tf.placeholder(tf.float32, [None, 10])

        # Define operations for testing
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # If provided, restore from a previously saved model
        if restore_from:
            tf.train.Saver().restore(session, restore_from)

        # Test against all of the MNIST test data
        test_accuracy = session.run(accuracy, feed_dict={self.input_layer: mnist.test.images,
                                                         labels: mnist.test.labels,
                                                         self.is_training: test_training_accuracy})
        print('-'*75)
        print('{}: Accuracy on full test set = {}'.format(self.name, test_accuracy))

        # If requested, perform tests predicting individual values rather than batches
        if include_individual_predictions:
            predictions = []
            correct = 0

            # Do 200 predictions, 1 at a time
            for i in range(200):
                # This is a normal prediction using an individual test case. However, notice
                # we pass `test_training_accuracy` to `feed_dict` as the value for `self.is_training`.
                # Remember that will tell it whether it should use the batch mean & variance or
                # the population estimates that were calucated while training the model.
                pred, corr = session.run([tf.arg_max(self.output_layer, 1), accuracy],
                                         feed_dict={self.input_layer: [mnist.test.images[i]],
                                                    labels: [mnist.test.labels[i]],
                                                    self.is_training: test_training_accuracy})
                correct += corr

                predictions.append(pred[0])

            print("200 Predictions:", predictions)
            print("Accuracy on 200 samples:", correct/200)
