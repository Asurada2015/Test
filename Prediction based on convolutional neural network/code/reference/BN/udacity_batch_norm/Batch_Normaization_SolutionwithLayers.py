# Batch Normalization – Solutions
# Batch Normalization 解决方案
"""
批量标准化在构建深度神经网络时最为有用。为了证明这一点，我们将创建一个具有20个卷积层的卷积神经网络，然后是一个完全连接的层。
我们将使用它来对MNIST数据集中的手写数字进行分类，现在您应该熟悉这一点。这不是划分MNIST数字的最好网络。您可以创建更简单的网络并获得更好的结果。
但是，为了给您批量标准化的实践经验，我们将使用这个作为一个例子:
1:这个网络足够复杂，可以保证体现BN算法对深层神经网络进行训练时的优势
2:这个例子比较简单，你可以很快获得训练的结果，这个简短的练习只是为了给你一次向深度神经玩过中添加BN算法的机会
3:足够简单，无需额外资源即可轻松理解架构。
"""
# 这个教程中有两种你可以自行编辑的在CNN中实现Batch Normalization的方法,
# 第一个是使用高级函数'tf.layers.batch_normalization',
# 第二个使用低级函数'tf.nn.batch_normalization'

# 下载MNIST手写数字识别数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

# Batch Normalization using tf.layers.batch_normalization
# 使用tf.layers.batch_normalization实现Batch Normalization
"""
这个版本的神经网络代码使用tf.layers包来编写，也推荐你使用tf.layers包函数来实现CNN和Batch Normalization算法。
我们将使用以下函数在我们的网络中创建完全连接的层。我们将用指定数量的神经元和ReLU激活函数来创建它们。
PS：这个版本的函数不包括批量标准化。
"""


def fully_connected(prev_layer, num_units):
    """
    num_units参数传递该层神经元的数量，根据prev_layer参数传入值作为该层输入创建全连接神经网络。
    :param prev_layer: Tensor
        该层神经元输入
    :param num_units: int
        该层神经元结点个数
    :returns Tensor
        一个新的全连接神经网络层
    """
    layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.relu)
    return layer


"""
我们会运用以下方法来构建神经网络的卷积层，这个卷积层很基本，我们总是使用3x3内核，ReLU激活函数，
在具有奇数深度的图层上步长为1x1，在具有偶数深度的图层上步长为2x2。在这个网络中，我们并不打算使用池化层。
PS：该版本的函数不包括批量标准化操作。
"""


def conv_layer(prev_layer, layer_depth):
    """
    Create a convolutional layer with the given layer as input.
    使用给定的参数作为输入创建卷积层
    :param prev_layer: Tensor
        传入该层神经元作为输入
    :param layer_depth: int
        我们将根据网络中图层的深度设置特征图的步长和数量。
        这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
    :returns Tensor
        一个新的卷积层
    """
    strides = 2 if layer_depth%3 == 0 else 1
    conv_layer = tf.layers.conv2d(prev_layer, layer_depth*4, 3, strides, 'same', activation=tf.nn.relu)
    return conv_layer


# 建立没有批量标准化的网络，然后在MNIST数据集上进行训练。它在训练期间定期显示Loss值和准确性数据


def train(num_batches, batch_size, learning_rate):
    # 为输入的样本和标签创建占位符
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # Feed the inputs into a series of 20 convolutional layers
    # 将输入数据填充到20个卷积层
    layer = inputs
    for layer_i in range(1, 20):
        layer = conv_layer(layer, layer_i)

    # Flatten the output from the convolutional layers
    # 将卷积层输出扁平化处理
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])

    # Add one fully connected layer
    # 添加一个具有100个神经元的全连接层
    layer = fully_connected(layer, 100)

    # Create the output layer with 1 node for each
    # 为每一个类别添加一个输出节点
    logits = tf.layers.dense(layer, 10)

    # 定义
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    # 训练和测试神经网络
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # train this batch
            # 训练批数据
            sess.run(train_opt, {inputs: batch_xs,
                                 labels: batch_ys})

            # Periodically check the validation or training loss and accuracy
            # 定期检查训练或验证集上的loss和精确度
            if batch_i%100 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: mnist.validation.images,
                                                              labels: mnist.validation.labels})
                print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            elif batch_i%25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        # 最后在验证集和测试集上对模型准确率进行评分
        acc = sess.run(accuracy, {inputs: mnist.validation.images,
                                  labels: mnist.validation.labels})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: mnist.test.images,
                                  labels: mnist.test.labels})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        # 对100个独立的测试图片进行评分,对比验证Batch Normalization的效果

        correct = 0
        for i in range(100):
            correct += sess.run(accuracy, feed_dict={inputs: [mnist.test.images[i]],
                                                     labels: [mnist.test.labels[i]]})

        print("Accuracy on 100 samples:", correct/100)


num_batches = 800  # 迭代次数
batch_size = 64  # 批处理数量
learning_rate = 0.002  # 学习率

tf.reset_default_graph()
with tf.Graph().as_default():
    train(num_batches, batch_size, learning_rate)

"""
有了这么多的层次，这个网络需要大量的迭代来学习。在您完成800个批次的培训时，您的最终测试和验证准确度可能不会超过10％。 
(每次都会有所不同，但很可能会低于15％)使用批量标准化，您可以在相同数量的批次中训练同一网络达到90％以上
我们将在Batch_Normaization_SolutionBNwithLayers进行详细介绍，使用tf.layers包构建带有BN层的卷积神经网络。
"""

# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# 2018-03-18 16:05:02.607870: I D:\Build\tensorflow\tensorflow-r1.4\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
# Batch:  0: Validation loss: 0.69079, Validation accuracy: 0.10700
# Batch: 25: Training loss: 0.33298, Training accuracy: 0.10938
# Batch: 50: Training loss: 0.32532, Training accuracy: 0.07812
# Batch: 75: Training loss: 0.32597, Training accuracy: 0.09375
# Batch: 100: Validation loss: 0.32531, Validation accuracy: 0.11260
# Batch: 125: Training loss: 0.32369, Training accuracy: 0.15625
# Batch: 150: Training loss: 0.32454, Training accuracy: 0.12500
# Batch: 175: Training loss: 0.32519, Training accuracy: 0.14062
# Batch: 200: Validation loss: 0.32540, Validation accuracy: 0.10700
# Batch: 225: Training loss: 0.32509, Training accuracy: 0.06250
# Batch: 250: Training loss: 0.32508, Training accuracy: 0.10938
# Batch: 275: Training loss: 0.32465, Training accuracy: 0.14062
# Batch: 300: Validation loss: 0.32541, Validation accuracy: 0.11260
# Batch: 325: Training loss: 0.32266, Training accuracy: 0.15625
# Batch: 350: Training loss: 0.32408, Training accuracy: 0.06250
# Batch: 375: Training loss: 0.32685, Training accuracy: 0.10938
# Batch: 400: Validation loss: 0.32567, Validation accuracy: 0.10020
# Batch: 425: Training loss: 0.32492, Training accuracy: 0.12500
# Batch: 450: Training loss: 0.32439, Training accuracy: 0.12500
# Batch: 475: Training loss: 0.32574, Training accuracy: 0.12500
# Batch: 500: Validation loss: 0.32554, Validation accuracy: 0.09860
# Batch: 525: Training loss: 0.32668, Training accuracy: 0.03125
# Batch: 550: Training loss: 0.32549, Training accuracy: 0.03125
# Batch: 575: Training loss: 0.32473, Training accuracy: 0.12500
# Batch: 600: Validation loss: 0.32628, Validation accuracy: 0.11260
# Batch: 625: Training loss: 0.32547, Training accuracy: 0.09375
# Batch: 650: Training loss: 0.32518, Training accuracy: 0.17188
# Batch: 675: Training loss: 0.32284, Training accuracy: 0.15625
# Batch: 700: Validation loss: 0.32541, Validation accuracy: 0.10700
# Batch: 725: Training loss: 0.32801, Training accuracy: 0.06250
# Batch: 750: Training loss: 0.32847, Training accuracy: 0.06250
# Batch: 775: Training loss: 0.32251, Training accuracy: 0.20312
# Final validation accuracy: 0.11260
# Final test accuracy: 0.11350
# Accuracy on 100 samples: 0.14
