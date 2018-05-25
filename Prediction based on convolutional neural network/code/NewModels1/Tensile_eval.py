import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import os

ops.reset_default_graph()

sess = tf.Session()

# 设置模型超参数


image_height = 20  # 图片高度
image_width = 20  # 图片宽度
num_channels = 1  # 图片通道数
num_targets = 1  # 预测指标数
MIN_AFTER_DEQUEUE = 100  # 管道最小容量
MODEL_SAVE_PATH = './Tensile_log'
MODEL_NAME = 'model.ckpt'
NUM_THREADS = 1  # 线程数
TRAIN_FILE = '235b_train_1.csv'
TEST_FILE = '235b_test_1.csv'
EVAL_FILE = '235b_test_1.csv'
save_eval_file = 'eval.csv'

# 自适应学习率衰减
eval_epoch = 1
eval_batch = 10


# RandomShuffleQueue '_1_shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested 3000, current size 1680)

# 读取数据
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.]]
    C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, B, Furnace, RoughMill, FinishMill, DownCoil, Tensile, Yeild, Elongation \
        = tf.decode_csv(value, defaults)
    vertor_example = tf.stack(
        [C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, B, Furnace, RoughMill, FinishMill,
         DownCoil])
    # 将(20)维度的数据添加维度成为(1,20)的向量

    example_2D = tf.expand_dims(vertor_example, 0)
    trans_example_2D = tf.transpose(example_2D)
    example = tf.expand_dims(tf.matmul(trans_example_2D, example_2D), 2)
    vertor_label = tf.stack([Tensile])
    return example, vertor_label


# 创建输入管道
def create_pipeline(filename, batch_size, num_threads, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)  # 设置文件名队列
    example, label = read_data(file_queue)  # 读取数据和标签

    min_after_dequeue = MIN_AFTER_DEQUEUE
    # capacity = min_after_dequeue + batch_size
    capacity = min_after_dequeue + (num_threads + 3*batch_size)
    example_batch, label_batch = tf.train.batch(
        [example, label], batch_size=batch_size, num_threads=num_threads, capacity=capacity)
    return example_batch, label_batch


# 定义模型架构

def inference(input_images, batch_size, is_training):
    # 截断高斯函数初始化
    def truncated_normal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype,
                                initializer=tf.truncated_normal_initializer(stddev=0.05)))

    # 0初始化
    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

    # 第一卷积层
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(input_images, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        relu_conv1 = tf.nn.relu(conv1, name='relu_conv1')
    # 池化层
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer1')

    # 第二个卷积层
    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(pool1, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        relu_conv2 = tf.nn.relu(conv2, name='relu_conv2')

    # 池化层/下采样层
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer2')

    # 第三个卷积层
    with tf.variable_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(pool2, 128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        relu_conv3 = tf.nn.relu(conv3, name='relu_conv3')

    # 池化层/下采样层
    pool3 = tf.nn.max_pool(relu_conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer3')

    # 光栅化处理，将其打平方便和全连接层进行连接
    reshaped_output = tf.reshape(pool3, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # 第四个卷积层
    with tf.variable_scope('conv4') as scope:
        conv4 = tf.layers.conv2d(pool3, 128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        relu_conv4 = tf.nn.relu(conv4, name='relu_conv4')

    # 池化层/下采样层
    pool4 = tf.nn.max_pool(relu_conv4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer4')

    # 第五个卷积层
    with tf.variable_scope('conv5') as scope:
        conv5 = tf.layers.conv2d(pool4, 256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        relu_conv5 = tf.nn.relu(conv5, name='relu_conv4')

    # 池化层/下采样层
    pool5 = tf.nn.max_pool(relu_conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer4')

    # 光栅化处理，将其打平方便和全连接层进行连接
    reshaped_output = tf.reshape(pool5, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # 全连接层1
    with tf.variable_scope('full1') as scope:
        full_layer1 = tf.layers.dense(reshaped_output, 256, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer1 = tf.layers.batch_normalization(full_layer1, training=is_training)
        full_layer1 = tf.nn.relu(full_layer1)

    # 全连接层2
    with tf.variable_scope('full2') as scope:
        # 第二个全连接层有192个输出
        full_layer2 = tf.layers.dense(full_layer1, 256, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer2 = tf.layers.batch_normalization(full_layer2, training=is_training)
        full_layer2 = tf.nn.relu(full_layer2)
    #
    # 全连接层3
    with tf.variable_scope('full3') as scope:
        # 第二个全连接层有192个输出
        full_layer3 = tf.layers.dense(full_layer2, 128, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer3 = tf.layers.batch_normalization(full_layer3, training=is_training)
        full_layer3 = tf.nn.relu(full_layer3)

    # 全连接层4
    with tf.variable_scope('full4') as scope:
        # 第二个全连接层有192个输出
        full_layer4 = tf.layers.dense(full_layer3, 64, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer4 = tf.layers.batch_normalization(full_layer4, training=is_training)
        full_layer4 = tf.nn.relu(full_layer4)

    # # 最后的全连接层只有1个输出
    # with tf.variable_scope('full5') as scope:
    #     full_weight5 = truncated_normal_var(name='full_mult5', shape=[64, num_targets], dtype=tf.float32)
    #     full_bias5 = zero_var(name='full_bias5', shape=[num_targets], dtype=tf.float32)
    #     final_output = tf.add(tf.matmul(full_layer4, full_weight5), full_bias5)

    # 测试只有一层全连接层时使用
    with tf.variable_scope('full') as scope:
        full_weight = truncated_normal_var(name='full_mult', shape=[64, num_targets], dtype=tf.float32)
        full_bias = zero_var(name='full_bias5', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer4, full_weight), full_bias)
    return (final_output)


# 获取数据
eval_images, eval_targets = create_pipeline(EVAL_FILE, batch_size=eval_batch, num_threads=3,
                                            num_epochs=eval_epoch)
# 声明模型
is_training = tf.placeholder(tf.bool)
with tf.variable_scope('model_definition') as scope:
    eval_output = inference(eval_images, batch_size=eval_batch, is_training=False)

# 初始化变量和队列
print('Initializing the Variables.')
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
sess.run([init_op, local_init_op])
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

saver = tf.train.Saver()
# tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    # 加载模型
    saver.restore(sess, ckpt.model_checkpoint_path)
    # 通过文件名得到模型保存是迭代的轮数
    # print('eval output')
    # Eval_output = sess.run(eval_output)
    # print(Eval_output)
    for i in range(10):
        print('the {} batch of data '.format(i+1))
        print(sess.run(eval_targets))
else:
    print('No checkpoint file found')

# 保存预测模型输出的值
# log_eval = []
# for i in Eval_output:
#     log_eval.append(i)
# with open(save_eval_file, "w", newline='') as f:
#     writer = csv.writer(f)
#     for a in range(Eval_output.__len__()):
#         writer.writerows([log_eval[a]])
# f.close()

# 关闭线程和Session
coord.request_stop()
coord.join(threads)
sess.close()
