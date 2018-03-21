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
            # Batch normalization 和平时一样使用权值，但是不用使用偏置项，这时我们需要额外计算gamma和beta这两个额外项而不用使用偏置属性
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
            # 当不使用Batch Normalization时，创建一个使用权值和输入相乘后加上偏置的标准层然后选择性的添加激活函数
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
        :参数session: Session
            用于运行测试图模型
        :param test_training_accuracy: bool (default False)
            If True, perform inference with batch normalization using batch mean and variance;
            if False, perform inference with batch normalization using estimated population mean and variance.
            Note: in real life, *always* perform inference using the population mean and variance.
                  This parameter exists just to support demonstrating what happens if you don't.
        :参数 test_training_accuracy: bool (default False)
        如果选择True,则会使用批数据中的平均值和方差使用Batch Normalization 算法构建前向传播预测
        如果选择False,则会使用估计的平均值和方差来使用Batch Normalization 构建前向传播
        注意：在实际生产中，一般使用估计的平均值和方差来进行预测数据上的前向传播，这里存在这个参数只是为了支持如果不用估计的平均值和方差的情况
        :param include_individual_predictions: bool (default True)
            This function always performs an accuracy test against the entire test set. But if this parameter
            is True, it performs an extra test, doing 200 predictions one at a time, and displays the results
            and accuracy.
        :参数 include_individual_predictions: bool (default True)
        此功能始终对整个测试集执行精度测试。但是，如果此参数为“真”，则会执行额外的测试，一次执行200个预测，并显示结果和准确性

        :param restore_from: string or None (default None)
            Name of a saved model if you want to test with previously saved weights.
        ：参数 restore_from: string or None (default None)
        如果要使用先前保存的权重进行测试，则可以传入模型的名称作为参数。
        """
        # This placeholder will store the true labels for each mini batch
        # 这个占位符将存储每个小批量的真实标签
        labels = tf.placeholder(tf.float32, [None, 10])

        # Define operations for testing
        # 定义测试使用的函数
        correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # If provided, restore from a previously saved model
        # 如果提供这个参数，我们会从先前保存的模型中重新加载模型
        if restore_from:
            tf.train.Saver().restore(session, restore_from)

        # Test against all of the MNIST test data
        # 在所有MNIST测试图片上进行测试
        test_accuracy = session.run(accuracy, feed_dict={self.input_layer: mnist.test.images,
                                                         labels: mnist.test.labels,
                                                         self.is_training: test_training_accuracy})
        print('-'*75)
        print('{}: Accuracy on full test set = {}'.format(self.name, test_accuracy))

        # If requested, perform tests predicting individual values rather than batches
        # 如果给出了include_individual_predictions参数，在独立的测试数据上测试而不是整个批次数据
        if include_individual_predictions:
            predictions = []
            correct = 0

            # Do 200 predictions, 1 at a time
            for i in range(200):
                # This is a normal prediction using an individual test case. However, notice
                # we pass `test_training_accuracy` to `feed_dict` as the value for `self.is_training`.
                # Remember that will tell it whether it should use the batch mean & variance or
                # the population estimates that were calucated while training the model.
                # 这是使用单个测试用例的正常预测。 但是，请注意我们将`test_training_accuracy`传递给`feed_dict`
                # 作为`self.is_training`的值。 请记住，它会告诉它是否应该使用批处理平均值和方差，或者是训练模型时计算出的估计值。
                pred, corr = session.run([tf.argmax(self.output_layer, 1), accuracy],
                                         feed_dict={self.input_layer: [mnist.test.images[i]],
                                                    labels: [mnist.test.labels[i]],
                                                    self.is_training: test_training_accuracy})
                correct += corr

                predictions.append(pred[0])

            print("200 Predictions:", predictions)
            print("Accuracy on 200 samples:", correct/200)


# 代码中有相当多的注释，所以这些应该回答你的大部分问题。
# 但是，让我们来看看最重要的几行。
# 我们将批规范化添加到fully_connected函数内的图层。以下是关于该代码的一些重要观点：
"""
1.使用Batch Normalization的神经层不包括偏置项
2.我们使用tf.layers.batch_normalization函数来解决数学问题(我们之后将会实现不使用这个算法的第低级版本)
3.我们告诉tf.layers.batch_normalization网络是否正在训练。这是我们稍后谈论的重要一步。
4.我们在调用激活函数之前使用BN规范化


注意：我们除了使用tf.layers.batch_normalization函数之外，我们还需要使用
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
作为声明。
这一行实际上与我们传递给tf.layers.batch_normalization的训练参数结合使用，没有它，Tensorflow的BN层将在前向传播时无法正确运行


最后，无论我们时训练网络或者执行前向传播推理，我们都要使用feed_dict将self.is_training 分别设置为True或False
eg:session.run(train_step, feed_dict={self.input_layer: batch_xs, 
                                               labels: batch_ys, 
                                               self.is_training: True})
                                               
稍后我们会详细介绍这些代码，但接下来我们要展示一些使用此代码的实验，以及使用和不使用Batch Normalization进行测试的网络
"""

# Batch Normalization 使用例子
"""
笔记的这一部分将通过训练各种有无Batch Normalization的网络来演示前面提到的一些好处。
我们要感谢这篇博文的作者在Tensorflow中实现Batch Normalization[https://r2rt.com/implementing-batch-normalization-in-tensorflow.html]
这篇文章提供了一些想法和一些代码来绘制训练期间准确度的差异和使用相同的权重来初始化多个不同网络的想法。
"""

# 支持测试的代码
"""
以下两个函数方法支持我们在notebook中运行的例子

第一个函数 plot_training_accuracies,简单的绘制了NeuralNet类传递给training_accuracies列表的精确度数值
如果你看了NeuralNet中的训练功能，你会发现它在训练网络时会定期测量验证的准确性并将结果存储在该列表中。

第二个函数train_and_test创建了两个神经网络 一个使用了Batch Normalization和一个没有Batch Normalization。然后训练它们并对它们进行测试，
调用plot_training_accuracies函数来绘制它们在训练过程中的准确度如何变化。train_and_test函数真正重要的一点是，它初始化网络外部神经元层的起始权重，然后将它们传入。
这使得它可以从完全相同的起始权重训练两个网络，这消除了由不同初始权重导致的性能差异。
"""


def plot_training_accuracies(*args, **kwargs):
    """
    Displays a plot of the accuracies calculated during training to demonstrate
    how many iterations it took for the model(s) to converge.
    显示训练期间计算的精度图，以显示模型收敛需要多少次迭代。

    :param args: One or more NeuralNet objects
        You can supply any number of NeuralNet objects as unnamed arguments
        and this will display their training accuracies. Be sure to call `train`
        the NeuralNets before calling this function.
    :参数 args: One or more NeuralNet objects
    你可以提供任意数量的NeuralNet对象作为未命名的参数，这将显示他们的训练精度。在调用这个函数之前，一定要调用`训练'NeuralNets'对象。

    :param kwargs:
        You can supply any named parameters here, but `batches_per_sample` is the only
        one we look for. It should match the `batches_per_sample` value you passed
        to the `train` function.
    :param kwargs:
    你可以在这里提供任何命名参数，但`batches_per_sample`是唯一的我们寻找的一个。
    它应该与您传递的`batches_per_sample`值相匹配到"训练"功能。
    """
    fig, ax = plt.subplots()

    batches_per_sample = kwargs['batches_per_sample']

    for nn in args:
        ax.plot(range(0, len(nn.training_accuracies)*batches_per_sample, batches_per_sample),
                nn.training_accuracies, label=nn.name)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy During Training')
    ax.legend(loc=4)
    ax.set_ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()


def train_and_test(use_bad_weights, learning_rate, activation_fn, training_batches=50000, batches_per_sample=500):
    """
    Creates two networks, one with and one without batch normalization, then trains them
    with identical starting weights, layers, batches, etc. Finally tests and plots their accuracies.
    创建两个神经网络，一个使用Batch Normalization 一个不使用Batch Normalization
    然后使用相同的初始权重，层，批次量等等，最后检测并画出精确度曲线

    :param use_bad_weights: bool
        If True, initialize the weights of both networks to wildly inappropriate weights;
        if False, use reasonable starting weights.
    :参数 use_bad_weights: bool
        如果为True，则将两个网络的权重初始化为非常不恰当的权重;
        如果是False，则使用合理的起始权重。

    :param learning_rate: float
        Learning rate used during gradient descent.
    :参数 学习率：float
        梯度下降时使用的学习率

    :param activation_fn: Callable
        The function used for the output of each hidden layer. The network will use the same
        activation function on every hidden layer and no activate function on the output layer.
        e.g. Pass tf.nn.relu to use ReLU activations on your hidden layers.
     :参数 激活函数: 可调用
        对隐藏层输出执行的函数，在每一个隐层会使用相同的激活函数，但是在最后一层不使用激活函数。
        eg: 通过传入tf.nn.relu在隐藏层使用ReLU激活函数

    :param training_batches: (default 50000)
        Number of batches to train.
     :参数 training_batches: (default 50000)
        训练批次数
    :param batches_per_sample: (default 500)
     :参数 batches_per_sample: (default 500)
        How many batches to train before sampling the validation accuracy.
        每训练batches_per_sample后使用验证集计算准确率
    """
    # Use identical starting weights for each network to eliminate differences in
    # weight initialization as a cause for differences seen in training performance
    # 为每个网络使用相同的起始权重以消除权重初始化中的差异，这因为权重初始化是造成培训性能差异的一个重要原因
    # Note: The networks will use these weights to define the number of and shapes of
    #       its layers. The original batch normalization paper used 3 hidden layers
    #       with 100 nodes in each, followed by a 10 node output layer. These values
    #       build such a network, but feel free to experiment with different choices.
    #       However, the input size should always be 784 and the final output should be 10.
    # 注意：网络将使用这些权重来定义神经网络模型的形状,原始Batch Normalization模型使使用3个隐藏层
    #      每个节点有100个节点，后面是10个节点输出层。你也可以建立这样一个网络，并随时尝试不同的选择。
    #      但是，输入大小应始终为784，最终输出应为10
    if use_bad_weights:
        # These weights should be horrible because they have such a large standard deviation
        # 这些数值很坏因为他们具有十分大的标准差
        weights = [np.random.normal(size=(784, 100), scale=5.0).astype(np.float32),
                   np.random.normal(size=(100, 100), scale=5.0).astype(np.float32),
                   np.random.normal(size=(100, 100), scale=5.0).astype(np.float32),
                   np.random.normal(size=(100, 10), scale=5.0).astype(np.float32)
                   ]
    else:
        # These weights should be good because they have such a small standard deviation
        # 这些初始化权重很好，因为它们具有很小的标准差
        weights = [np.random.normal(size=(784, 100), scale=0.05).astype(np.float32),
                   np.random.normal(size=(100, 100), scale=0.05).astype(np.float32),
                   np.random.normal(size=(100, 100), scale=0.05).astype(np.float32),
                   np.random.normal(size=(100, 10), scale=0.05).astype(np.float32)
                   ]

    # Just to make sure the TensorFlow's default graph is empty before we start another
    # test, because we don't bother using different graphs or scoping and naming
    # elements carefully in this sample code.
    # 在我们开始另一个测试时我们要确保Tensorflow的默认图模型是空的，因为我们不会在这个示例代码中仔细地使用不同的图或范围来命名元素。
    tf.reset_default_graph()

    # build two versions of same network, 1 without and 1 with batch normalization
    # 建立两个不同版本的相同神经网络，一个使用Batch Normalization 另一个不使用
    nn = NeuralNet(weights, activation_fn, False)
    bn = NeuralNet(weights, activation_fn, True)

    # train and test the two models
    # 训练并测试两个模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        nn.train(sess, learning_rate, training_batches, batches_per_sample)
        bn.train(sess, learning_rate, training_batches, batches_per_sample)

        nn.test(sess)
        bn.test(sess)

    # Display a graph of how validation accuracies changed during training so we can compare
    #  how the models trained and when they converged
    # 显示培训期间验证准确度如何变化的图形，以便我们可以比较模型的训练和收敛时间
    plot_training_accuracies(nn, bn, batches_per_sample=batches_per_sample)


# 下一系列单元通过各种设置来训练网络，以显示有无Batch Normalization的差异。它们旨在清楚地表明批量标准化的效果。
# 下面使用ReLU激活函数创建两个网络，学习率为0.01，并且启动权重合理。


# train_and_test(False, 0.01, tf.nn.relu)


# 100%|██████████| 50000/50000 [01:25<00:00, 587.98it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.9742000102996826
# 100%|██████████| 50000/50000 [02:29<00:00, 333.88it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.980400025844574
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.9739000201225281
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9786999821662903
"""
图片保存为:train_and_test_ReLU001_50000.png
如预期的那样，两个网络训练良好并最终达到相似的测试精度。但是，请注意，批量归一化模型的收敛速度比其他网络要快一些，
几乎立即达到90％以上的准确度，并在10或15000次迭代中接近最大准确度。另一个网络需要大约3千次迭代才能达到90％，并且在3万次或更多迭代之前不会达到最佳精度。

如果您查看原始速度，您可以看到，如果没有批量标准化，我们每秒计算超过1100次迭代，而批量标准化会降低到500个以上。
然而，批量标准化允许我们执行更少的迭代并且收敛时需要更少的时间。（我们只在这里培训了5万迭代，所以我们可以绘制比较结果。）
"""

# 以下示例创建了两个神经网络，它们使用前面示例中使用的相同的参数，但只有2000次迭代的训练
# train_and_test(False, 0.01, tf.nn.relu, 2000, 50)


# 图片保存为train_and_test_0012000.png
# 100%|██████████| 2000/2000 [00:04<00:00, 478.34it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.8378000259399414
# 100%|██████████| 2000/2000 [00:07<00:00, 273.43it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.953000009059906
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.8328999876976013
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9487000107765198
"""
正如你所看到的，使用批量标准化生成的模型只运行2000个迭代次数精度超过了95％，而在大约500个迭代的某处达到了90％以上。
如果没有批量标准化，模型需要1750次迭代才能达到80％ - 批量标准化的网络在经过大约200次迭代后就会达到该标记！ 
(注意：你自己在运行时，尽管模型每次的权重初始化相同，但是模型每次运行时生成的随机数不同)。 

在上面的例子中，你也应该注意到网络每秒训练的批次数少于你在前面的例子中看到的。
这是因为我们正在追踪的大部分时间实际上都是周期性地执行inference并收集用以显示的数据的过程。(inference是很花时间的)
在这个例子中，我们每50个批次执行一次inference，而不是每500个进行一次inference(前向传播预测)，
因此这个例子相对于上个模型生成2000次的图，时间的花销提高了十倍
因此为这个例子生成图需要相同2000次迭代的开销的10倍。
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为0.01，使用合理的起始权重。
# train_and_test(False, 0.01, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid00150000
# 100%|██████████| 50000/50000 [01:22<00:00, 608.17it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.8184000253677368
# 100%|██████████| 50000/50000 [02:26<00:00, 342.17it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9732000231742859
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.8094000220298767
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9732000231742859

"""
因为我们使用的图层数量和这个小的学习速率，使用S形激活函数需要很长时间才能开始学习。 
它最终开始取得进展，但为了获得超过80％的准确度，它花费了超过45,000批次。 
在大约一千个迭代批次中使用批量标准化可达到90％。
"""

# 下面使用ReLU激活函数创建两个网络，学习率为1，使用合理的起始权重。
# train_and_test(False, 1, tf.nn.relu)

# 图片保存为train_and_test_ReLU150000
# 100%|██████████| 50000/50000 [01:27<00:00, 568.55it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.0957999974489212
# 100%|██████████| 50000/50000 [02:40<00:00, 311.91it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9851999878883362
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.09799999743700027
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9846000075340271

"""
现在我们再次使用ReLUs，但学习速度更快。 该图显示了培训如何开始非常正常，批量规范化的网络开始比另一个更快。
但是更高的学习速度会使精度提高一点点，并且在某些情况下，没有批量标准化的网络的准确性就会完全崩溃。 
由于学习率较高，很可能过多的ReLU会在此死亡。

"""

# 下面使用S形激活函数创建两个网络，学习率为1，合理的起始权重。
# train_and_test(False, 1, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid1_50000
# 100%|██████████| 50000/50000 [01:31<00:00, 549.33it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.9750000238418579
# 100%|██████████| 50000/50000 [02:32<00:00, 327.02it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9805999994277954
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.9768999814987183
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.980400025844574

"""
在这个例子中，我们切换到一个sigmoid激活函数。 它似乎很好地处理了较高的学习率，两个网络都达到了高精度。

下面的单元显示了一个类似的网络对，只有2000次迭代训练。
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为1,2000迭代
# train_and_test(False, 1, tf.nn.sigmoid, 2000, 50)

# 图片保存为train_and_test_Sigmoid1_2000
# 100%|██████████| 2000/2000 [00:04<00:00, 435.24it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.8619999885559082
# 100%|██████████| 2000/2000 [00:07<00:00, 256.29it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9588000178337097
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.857200026512146
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9531999826431274

"""
正如你所看到的，尽管这些参数在两个网络中都能很好地工作，但批量标准化的参数在400个左右的批次中可以达到90％以上，而另一个则需要1700次迭代才能达到一样的结果。
当训练更大的网络时，这些差异会变得更加明显。
"""

# 下面使用Relu激活函数创建两个网络，学习率为2,50000次迭代
# train_and_test(False, 2, tf.nn.relu)

# 图片保存为train_and_test_ReLU2_50000
# 100%|██████████| 50000/50000 [01:32<00:00, 541.12it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.11259999871253967
# 100%|██████████| 50000/50000 [02:41<00:00, 308.66it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9850000143051147
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.11349999904632568
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9825999736785889

"""
当使用非常大的学习率时，批量标准化的网络训练良好，几乎可以立即管理98％的准确性。但是，没有规范化的网络根本就不学习。
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为2,50000次迭代
# train_and_test(False, 2, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid2_50000
# 100%|██████████| 50000/50000 [01:22<00:00, 603.50it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.9801999926567078
# 100%|██████████| 50000/50000 [02:27<00:00, 338.78it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.982200026512146
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.972599983215332
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9818999767303467

"""
使用具有较大学习速率的S形激活函数可以很好地适用于批处理标准化，也可以不用批处理标准化。 
但是，请看下面的图，我们用相同的参数训练模型，但只有2000次迭代。像往常一样，批量标准化可以让培训更快。
但是不使用批量标准化的Sigmoid函数模型得不到很好地结果。
"""

# 下面使用Sigmoid 激活函数创建两个网络，学习率为2,2000次迭代
# train_and_test(False, 2, tf.nn.sigmoid, 2000, 50)

# 图片保存为train_and_testSigmoid2_2000
# 100%|██████████| 2000/2000 [00:04<00:00, 475.72it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.948199987411499
# 100%|██████████| 2000/2000 [00:07<00:00, 271.75it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9562000036239624
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.9441999793052673
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9562000036239624

"""
在其余的例子中，我们使用了非常差的起始权重。也就是说，通常我们会使用接近于零的非常小的值。
然而，在这些例子中，我们选择了标准偏差为5的随机值。如果你真的在训练一个神经网络，你不会想这样做。
但是这些例子说明了批量规范化如何让你的网络更具弹性。
"""

# 下面使用ReLU激活函数创建两个网络，学习率为0.01，使用坏权重初始化，50000次迭代
# train_and_test(True, 0.01, tf.nn.relu)

# 图片保存为train_and_test_ReLU001_bad50000
# 100%|██████████| 50000/50000 [01:22<00:00, 604.35it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.0957999974489212
# 100%|██████████| 50000/50000 [02:38<00:00, 316.32it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.8022000193595886
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.09799999743700027
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.7979999780654907

"""
如图所示，没有批量标准化，网络从不学习任何东西。但通过批量标准化，它实际上学得很好，准确度达到了近80％。
起始权重明显伤害了网络，但您可以看到批量标准化在克服它们方面的表现如何。
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为0.01，使用坏权重初始化，50000次迭代
# train_and_test(True, 0.01, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid001_bad50000
# 100%|██████████| 50000/50000 [01:21<00:00, 614.89it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.3765999972820282
# 100%|██████████| 50000/50000 [02:24<00:00, 346.64it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.8353999853134155
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.3776000142097473
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.8442999720573425

"""
在不使用BN情况下，使用坏权值，使用Sigmoid函数比使用ReLU得到的效果要好一点，但是会需要训练更长的时间
"""

# 下面使用ReLU激活函数创建两个网络，学习率为1，使用坏权重初始化，50000次迭代
# train_and_test(True, 1, tf.nn.relu)

# 图片保存为train_and_test_ReLU1_bad50000
# 100%|██████████| 50000/50000 [01:21<00:00, 615.20it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.0957999974489212
# 100%|██████████| 50000/50000 [02:23<00:00, 347.94it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.8650000095367432
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.09799999743700027
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.8604999780654907

"""
这里使用的更高的学习率允许批量标准化的网络在约3万批次中超过90％。没有使用它的网络则没有学习到任何东西
但是使用bad权值会导致更大的随机性，也有可能出现使用BN还是什么都学不到的情况
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为1，使用坏权重初始化，50000次迭代
# train_and_test(True, 1, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid1_bad50000
# 100%|██████████| 50000/50000 [01:22<00:00, 605.26it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.8916000127792358
# 100%|██████████| 50000/50000 [02:30<00:00, 332.20it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9553999900817871
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.8884000182151794
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9509000182151794

"""
对于这种更高的学习速度，使用sigmoid比ReLUs更有效。
但是，您可以看到，如果没有批量标准化，网络需要很长时间，大量反弹，花费很长时间才能达到90％。'
批量规范化网络训练速度更快，似乎更稳定，并且达到更高的准确性。
"""

# 下面使用ReLU激活函数创建两个网络，学习率为2，使用坏权重初始化，50000次迭代
# train_and_test(True, 2, tf.nn.relu)

# 图片保存为train_and_test_ReLU2_bad50000
# 100%|██████████| 50000/50000 [01:22<00:00, 607.89it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.0957999974489212
# 100%|██████████| 50000/50000 [02:23<00:00, 348.61it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.8222000002861023
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.09799999743700027
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.8181999921798706

"""
我们已经看到，学习率更高的时候ReLUs不如S形，学习率更高的S形，而且我们使用的速率非常高。
正如预期的那样，如果没有批量标准化，网络根本就不会学习。但通过批量标准化，最终达到82％的准确度。
不过请注意，训练期间它的准确性如何反弹，这是因为学习速度实在太高，所以在使用BN后能够奏效，这是有点运气的。
"""

# 下面使用Sigmoid激活函数创建两个网络，学习率为2，使用坏权重初始化，50000次迭代
# train_and_test(True, 2, tf.nn.sigmoid)

# 图片保存为train_and_test_Sigmoid2_bad50000
# 100%|██████████| 50000/50000 [01:20<00:00, 620.48it/s]
# Without Batch Norm: After training, final accuracy on validation set = 0.9114000201225281
# 100%|██████████| 50000/50000 [02:28<00:00, 336.21it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9613999724388123
# ---------------------------------------------------------------------------
# Without Batch Norm: Accuracy on full test set = 0.9038000106811523
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9602000117301941

"""
在这种情况下，批量归一化网络训练速度更快，准确性更高。
同时，高速学习率使得没有规范化的网络在不规则的反弹，并且难以查超过90%以上准确率
"""

# Batch Normalization 不能解决所有问题
"""
BN算法并不神奇，并且不一定每一次都能起作用，权重初始化是随机的，再训练期间随机选择批次，所以你永远不知道训练会如何进行。
即使对于这些测试，我们对两个网络使用相同的初始权重，但每次运行时仍然会得到不同的权重。 
本节包括两个示例，它们显示批处理正常化完全无法解决时运行的示例。 
"""

# 下面使用ReLU激活功能创建两个网络，学习率为1，并且启动权重不好。
# train_and_test(True, 1, tf.nn.relu)

# 保存图片为train_and _test_ReLU1_bad50000_1

"""
当我们早些时候使用这些相同的参数时，我们看到批量规范化网络的验证准确度达到了92％。
这次我们使用不同的起始权重，使用与以前相同的标准偏差进行初始化，并且网络根本不学习。 
(请记住，如果网络始终猜测相同的值，则网络的准确度大约为10％) 
"""

# 下面使用ReLU激活功能创建两个网络，学习率为2，起始权重不好。
train_and_test(True, 2, tf.nn.relu)

# 保存图片为train_and _test_ReLU2_bad50000_1

"""
当我们早些使用这些参数和批量标准化进行训练时，我们达到了90％的验证准确度。然而，这一次网络开始几乎开始取得一些进展，但很快就会崩溃并停止学习。
注意：上述两个示例都使用了非常差的起始权重，以及过高的学习率。
虽然我们已经展示了批量标准化可以克服不良的价值，但我们并不意味着鼓励实际使用它们。本
笔记本中的示例旨在表明批量标准化可以帮助您的网络更好地训练。但是最后两个例子应该提醒你，你仍然想尝试使用良好的网络设计选择和合理的起始权重。
它还应该提醒你，即使在使用其他相同的体系结构时，每次尝试训练网络的结果都是随机的。
"""
