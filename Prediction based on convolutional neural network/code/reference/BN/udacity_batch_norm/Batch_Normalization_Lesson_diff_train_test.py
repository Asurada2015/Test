# 在训练和前向传播预测时为什么有区别
# 在原先使用tf.layers.batch_normalization函数的代码中，我们通过传入一个'training'参数告诉神经层我们是否正在训练网络
# eg:batch_normalized_output = tf.layers.batch_normalization(linear_output, training=self.is_training)
# 这就导致我们在feed_dict向神经网络中传值时需要提供一个self.is_training的参数，例如在NeuraNet的train方法中:
# session.run(train_step, feed_dict={self.input_layer: batch_xs,
#                                    labels: batch_ys,
#                                    self.is_training: True})
# 如果你仔细看不使用tf.layers.batch_normalization的低级实现代码(即使用tf.nn.batch_normalization)的代码中我们在训练和前向传播推导时都有
# 一些不同，但是这是怎样产生的呢？
# 首先，我们看看当训练和预测前向传播时没有区别会发生什么。
# 以下函数与之前的train_and_test类似，但是这次我们只测试一个网络，而不是绘制其准确性，我们对测试输入执行200次预测，一次预测输入一个预测数据。
# 我们可以使用test_training_accuracy参数来监测网络时在训练还是在预测模型（相当于将true或false传递给feed_dict中的is_training参数）

import tensorflow as tf
import tqdm
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data so we have something for our experiments
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 对于网络的构建使用NeuralNet类和Batch_Normalization_Lesson_with_tf_layers文件中的NeuralNet中的类一样
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
        此功能默认对整个测试集执行精度测试。但是，如果此参数为“真”，则会执行另一种的测试，执行200次预测一次预测一个数据，并显示结果和准确性

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


def batch_norm_test(test_training_accuracy):
    """
    :param test_training_accuracy: bool
        If True, perform inference with batch normalization using batch mean and variance;
        if False, perform inference with batch normalization using estimated population mean and variance.
    :参数 test_training_accuracy: bool
    如果此值为True,batch normaization中的inference函数使用批数据的平均值和方差
    如果此值为False,batch normaization中的inference函数使用滑动平均估计的平均值和方差
    """

    weights = [np.random.normal(size=(784, 100), scale=0.05).astype(np.float32),
               np.random.normal(size=(100, 100), scale=0.05).astype(np.float32),
               np.random.normal(size=(100, 100), scale=0.05).astype(np.float32),
               np.random.normal(size=(100, 10), scale=0.05).astype(np.float32)
               ]

    tf.reset_default_graph()

    # Train the model
    bn = NeuralNet(weights, tf.nn.relu, True)

    # First train the network
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        bn.train(sess, 0.01, 2000, 2000)

        bn.test(sess, test_training_accuracy=test_training_accuracy, include_individual_predictions=True)


# 此处我们使用test_training_accuracy的参数为true,这和我们在训练时使用的Batch Normalization参数一样，而不是使用测试时的参数

# batch_norm_test(True)

# 100%|██████████| 2000/2000 [00:06<00:00, 312.87it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9521999955177307
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9502000212669373
# 200 Predictions: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
# 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
# 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# Accuracy on 200 samples: 0.05

"""
你可以看到，神经网络每次都猜测得到一个相同的值。因为在训练过程中，使用Batch Normalization的网络根据批数据中的平均值和方差调整每一层的值。
我们在预测时使用的数据每次只输入一个值，意味着每个批次只有一个数据，所以这个批次数据中的平均值就是输入值，而方差为0.这意味着网络会把每一层的值标准化到0
(查看BN算法的公式我们可以发现一个数据批次中的值等于其平均值时，这个值会被标准化到0的原因)
所以我们最终得到的结果对于我们给网络的每一个输入都是一样的，因为每一层的数据和weights相乘后得到的值标准化后都会变为0。

注意：当我们再次运行上面这段代码时我们可能会得到一个和8不同的值。这是因为每次神经网络学习到的特定权值都不一样。
但是无论这个值变成多少，这200个数据输出的值会是同一个数字。

为了解决这个问题，神经网络不仅仅是标准化在每一层对该批次数据进行标准化，而是在整个分布上估计平均值和方差。所以在使用BN算法进行前向推导时，我们不使用整个批次数据上的平均值和方差。
而是使用在训练时计算得到的整个训练数据集上的平均值和方差的分布。

所以在下面的例子中，我们向函数中参数'test_training_accuracy'传入'False'值，这告诉网络，我们想要使用在训练时计算的整个数据集的平均值和方差分布运用前向传播预测。
"""

batch_norm_test(False)

# 100%|██████████| 2000/2000 [00:06<00:00, 308.11it/s]
# With Batch Norm: After training, final accuracy on validation set = 0.9603999853134155
# ---------------------------------------------------------------------------
# With Batch Norm: Accuracy on full test set = 0.9498999714851379
# 200 Predictions: [7, 2, 1, 0, 4, 1, 4, 9, 6, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3,
# 6, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2, 4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3, 7, 4, 6, 4, 3, 0, 7, 0,
# 2, 9, 1, 7, 3, 7, 9, 7, 7, 6, 2, 7, 8, 4, 7, 3, 6, 1, 3, 6, 9, 3, 1, 4, 1, 7, 6, 9, 6, 0, 5, 4, 9, 9, 2, 1, 9, 4, 8,
# 7, 3, 9, 7, 4, 4, 4, 9, 2, 5, 4, 7, 6, 4, 9, 0, 5, 8, 5, 6, 6, 5, 7, 8, 1, 0, 1, 6, 4, 6, 7, 3, 1, 7, 1, 8, 2, 0, 9,
#  9, 3, 5, 5, 1, 5, 6, 0, 3, 4, 4, 6, 5, 4, 6, 5, 4, 5, 1, 4, 4, 7, 2, 3, 2, 3, 1, 8, 1, 8, 1, 8, 5, 0, 8, 9, 2, 5, 0,
#  1, 1, 1, 0, 7, 0, 3, 1, 6, 4, 2]
# Accuracy on 200 samples: 0.96