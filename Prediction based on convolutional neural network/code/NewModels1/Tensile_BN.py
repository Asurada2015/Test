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

output_every = 50  # 训练输出间隔/控制图像标尺
generations = 5002  # 迭代次数 20000
eval_every = 50  # 测试输出间隔/控制图像标尺
image_height = 20  # 图片高度
image_width = 20  # 图片宽度
num_channels = 1  # 图片通道数
num_targets = 1  # 预测指标数
MIN_AFTER_DEQUEUE = 1000  # 管道最小容量
BATCH_SIZE = 128  # 批处理数量  128 test use 3
SAVEValue = 1000  # 保存模型各项参数值
save_test_file = 'testParameter.csv'
save_train_file = 'trainParameter.csv'
ViewGraph = 500
Savemodel = 5000
MODEL_SAVE_PATH = './Tensile_log'
MODEL_NAME = 'model.ckpt'
# 数据输入
NUM_EPOCHS = 8000  # 批次轮数
NUM_THREADS = 3  # 线程数
TRAIN_FILE = '235b_train_1.csv'
TEST_FILE = '235b_test_1.csv'

# 自适应学习率衰减
learning_rate = 0.1  # 初始学习率
lr_decay = 0.9  # 学习率衰减速度
num_gens_to_wait = 200  # 学习率更新周期


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
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, num_threads=num_threads, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
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


# 损失函数MSE
def cnn_loss(logits, targets):
    mse = tf.reduce_mean(tf.square(logits - targets), name='mse')  # 均方误差
    return mse


# 训练阶段函数
def train_step(loss_value, generation_num):
    # 自适应学习率递减
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                     num_gens_to_wait, lr_decay, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(model_learning_rate).minimize(loss_value)
    return train_opt


# 计算R系数
def R2(logits, targets):
    mean_targets = tf.reduce_mean(targets)
    # ss_tot总体平方和
    ss_tot = tf.reduce_sum(tf.square(tf.subtract(targets, mean_targets)))
    # ss_err残差平方和
    ss_err = tf.reduce_sum(tf.square(tf.subtract(logits, targets)))
    r2 = 1 - (tf.truediv(ss_err, ss_tot))
    return r2


# 三种指标的R系数
def R2_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile = tf.unstack(tf.cast(trans_targets, tf.double))
    r2_Tensile = R2(logits_Tensile, targets_Tensile)
    return r2_Tensile


# 计算RMSE均方根误差
def RMSE(logits, targets):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, targets))))  # 均方根误差
    return rmse


# 三种指标的RMSE系数
def RMSE_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile = tf.unstack(tf.cast(trans_targets, tf.double))
    rmse_Tensile = RMSE(logits_Tensile, targets_Tensile)
    return rmse_Tensile


# 计算RPD值
def RPD(logits, targets):
    mean_of_targets = tf.reduce_mean(targets)
    stdev = tf.sqrt(tf.divide(tf.reduce_sum(tf.square(tf.subtract(targets, mean_of_targets))),
                              tf.cast((BATCH_SIZE - 1), tf.double)))  # 测定值标准差
    rmse = RMSE(logits, targets)  # 测定值均方误差
    rpd = tf.divide(stdev, rmse)
    return rpd


# 测试集上三种指标相对分析误差值RPD
def RPD_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile = tf.unstack(tf.cast(trans_targets, tf.double))
    rpd_Tensile = RPD(logits_Tensile, targets_Tensile)
    return rpd_Tensile


# 获取数据
print('Now getting and transforming Data')
train_images, train_targets = create_pipeline(TRAIN_FILE, batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                                              num_epochs=NUM_EPOCHS)

test_images, test_targets = create_pipeline(TEST_FILE, batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                                            num_epochs=NUM_EPOCHS)

# 声明模型
print('Creating the CNN model.')
is_training = tf.placeholder(tf.bool)
with tf.variable_scope('model_definition') as scope:
    model_output = inference(train_images, BATCH_SIZE, is_training)
    # 这非常重要，我们必须设置scope重用变量
    # 否则，当我们设置测试网络模型，它会设置新的随机变量，这会使在测试批次上进行随机评估，影响评估结果
    scope.reuse_variables()
    test_output = inference(test_images, BATCH_SIZE, is_training)
# 声明损失函数
print('Declare Loss Function.')
loss = cnn_loss(model_output, train_targets)
loss_in_testdata = cnn_loss(test_output, test_targets)

# 创建训练操作
print('Create the Train Operation')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# 初始化变量和队列
print('Initializing the Variables.')
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
sess.run([init_op, local_init_op])
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# 训练CNN模型
print('Starting Training')
train_lossl = []
train_r2_tl = []
train_rmse_tl = []
test_lossl = []
test_r2_tl = []
test_rmse_tl = []
test_rpd_tl = []

log_train = []
for a in range(3):
    log_train.append([])
log_test = []
for b in range(4):
    log_test.append([])

saver = tf.train.Saver()
for i in tqdm.tqdm(range(generations)):
    # _, loss_value = sess.run([train_op, loss], {is_training: True})
    # 根据Udacity中Batch normalization的教程，除了在梯度下降的时候使用is_training:True其余时候均将其设置为False
    # 显示在训练集上各项指标
    # 因为在前100次迭代过程中会有明显的下降过程不能分清细节
    sess.run(train_op, {is_training: True})
    if i >= 100:
        if (i + 1)%output_every == 0:
            train_R2_T = sess.run(R2_of_batch(model_output, train_targets), {is_training: False})
            train_RMSE_T = sess.run(RMSE_of_batch(model_output, train_targets),
                                    {is_training: False})
            loss_value = sess.run(loss, {is_training: False})
            train_lossl.append(loss_value)
            output = 'Generation {}: train Loss = {:.5f}'.format((i + 1), loss_value)
            train_r2_tl.append(train_R2_T)
            train_rmse_tl.append(train_RMSE_T)
            log_train[0].append(loss_value)
            log_train[1].append(train_R2_T)
            log_train[2].append(train_RMSE_T)
            print(output)  # 显示训练集上的loss值

        # 显示在测试集上各项指标

        if (i + 1)%eval_every == 0:
            test_R2_T = sess.run(R2_of_batch(test_output, test_targets), {is_training: False})
            test_RMSE_T = sess.run(RMSE_of_batch(test_output, test_targets),
                                   {is_training: False})
            test_loss_value = sess.run(loss_in_testdata, {is_training: False})
            test_RPD_T = sess.run(RPD_of_batch(test_output, test_targets), {is_training: False})
            test_lossl.append(test_loss_value)
            test_loss_output = 'Generation {}: test Loss = {:.5f}'.format((i + 1), test_loss_value)
            test_r2_tl.append(test_R2_T)
            test_rmse_tl.append(test_RMSE_T)
            test_rpd_tl.append(test_RPD_T)
            log_test[0].append(test_loss_value)
            log_test[1].append(test_R2_T)
            log_test[2].append(test_RMSE_T)
            log_test[3].append(test_RPD_T)
            print(test_loss_output)
            # 保存所有属性值
        if (i + 1)%SAVEValue == 0:
            save_Totest_file = str(i + 1) + save_test_file
            with open(save_Totest_file, "w", newline='') as f:
                writer = csv.writer(f)
                for a in range(4):
                    writer.writerows([log_test[a]])
            f.close()
            save_Totrain_file = str(i + 1) + save_train_file
            with open(save_Totrain_file, "w", newline='') as f:
                writer = csv.writer(f)
                for a in range(3):
                    writer.writerows([log_train[a]])
            f.close()

        if (i + 1)%ViewGraph == 0:
            # 打印损失函数
            output_indices = range(100, i + 1, output_every)
            eval_indices = range(100, i + 1, eval_every)

            # 显示训练集/测试集loss函数
            plt.plot(output_indices, train_lossl, label='loss in train dataset', linewidth=1.0, color='red',
                     linestyle='--')
            plt.plot(eval_indices, test_lossl, label='loss in test dataset', linewidth=1.0, color='blue')
            plt.title(' Loss of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('Loss')
            plt.legend(loc=1, fancybox=True, shadow=True)
            plt.savefig('loss_' + str(i + 1) + '.png', dpi=300)
            plt.show()
            #
            # 显示训练集/测试集R2函数变化
            plt.figure()
            plt.plot(output_indices, train_r2_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_r2_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' R of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('R')
            plt.legend(loc=4, fancybox=True, shadow=True)
            plt.savefig('R_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示训练集/测试集RMSE函数变化
            plt.figure()
            plt.plot(output_indices, train_rmse_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_rmse_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' RMSE of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('RMSE')
            plt.legend(loc=1, fancybox=True, shadow=True)
            plt.savefig('RMSE_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示测试集RPD函数变化
            plt.figure()
            plt.plot(eval_indices, test_rpd_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' RPD of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('RPD')
            plt.legend(loc=2, fancybox=True, shadow=True)
            plt.savefig('RPD_' + str(i + 1) + '.png', dpi=300)
            plt.show()
            plt.close()
            print('训练集上一个批次数据的前10个数据\n', sess.run(train_targets)[:10])
            print('测试集上一个批次数据的前10个数据\n', sess.run(test_targets)[:10])
            print('训练集上一个批次数据通过inference的前10个输出结果\n', sess.run(model_output, {is_training: False})[:10])
            print('测试集上一个批次数据中通过inference后的前10个输出结果\n', sess.run(test_output, {is_training: False})[:10])
        if i%Savemodel == 0:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
# 关闭线程和Session
coord.request_stop()
coord.join(threads)
sess.close()
