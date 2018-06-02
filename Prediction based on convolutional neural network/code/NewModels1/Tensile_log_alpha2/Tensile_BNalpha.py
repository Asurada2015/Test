import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.python.framework import ops
import csv
import os

ops.reset_default_graph()

sess = tf.Session()
"""alpha版的目的在于寻找最好的模型savemodel参数会去掉而使用保存最好的模型值来取代,所以每一次都会运行所有指标"""
# 设置模型超参数

output_every = 50  # 输出间隔/控制图像标尺
generations = 100002  # 迭代次数 20000
# eval_every = 1  # 测试输出间隔/控制图像标尺
image_height = 20  # 图片高度
image_width = 20  # 图片宽度
num_channels = 1  # 图片通道数
num_targets = 1  # 预测指标数
MIN_AFTER_DEQUEUE = 1000  # 管道最小容量
BATCH_SIZE = 128  # 批处理数量  128 test use 3
SAVEValue = 10000  # 保存模型各项参数值
save_test_file = 'testParameter.csv'
save_train_file = 'trainParameter.csv'
ViewGraph = 2000
ViewDate = 100
MODEL_SAVE_PATH = './Tensile_log_alpha'
MODEL_NAME = 'model.ckpt'
MAX_Test_RMSE = 0.03  # 0.03
MAX_Test_MSE = 0.0015  # 0.001
MIN_Test_R = 0.93  # 0.8
MIN_Test_RPD = 3.8  # 2.5
INVALID_NUMBER = 100
MAX_Test_MAE = 0.021
MAX_Test_MAPE = 0.042
# 数据输入
NUM_EPOCHS = 8000  # 批次轮数
NUM_THREADS = 3  # 线程数
TRAIN_FILE = '235b_train_1.csv'
TEST_FILE = '235b_test_1.csv'

# 自适应学习率衰减
learning_rate = 0.1  # 初始学习率
lr_decay = 0.96  # 学习率衰减速度
num_gens_to_wait = 90  # 学习率更新周期 decay_steps：衰减次数，为样本总数/个次训练的batch大小，固定值11520/128=90


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
        relu_conv5 = tf.nn.relu(conv5, name='relu_conv5')

    # 池化层/下采样层
    pool5 = tf.nn.max_pool(relu_conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer5')

    # 第六个卷积层
    with tf.variable_scope('conv6') as scope:
        conv6 = tf.layers.conv2d(pool5, 256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
        relu_conv6 = tf.nn.relu(conv6, name='relu_conv6')

    # 池化层/下采样层
    pool6 = tf.nn.max_pool(relu_conv6, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_layer6')

    # 光栅化处理，将其打平方便和全连接层进行连接
    reshaped_output = tf.reshape(pool6, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # 全连接层1
    with tf.variable_scope('full1') as scope:
        full_layer1 = tf.layers.dense(reshaped_output, 512, activation=None, use_bias=False,
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
        full_layer3 = tf.layers.dense(full_layer2, 256, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer3 = tf.layers.batch_normalization(full_layer3, training=is_training)
        full_layer3 = tf.nn.relu(full_layer3)

    # 全连接层4
    with tf.variable_scope('full4') as scope:
        # 第二个全连接层有192个输出
        full_layer4 = tf.layers.dense(full_layer3, 128, activation=None, use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        full_layer4 = tf.layers.batch_normalization(full_layer4, training=is_training)
        full_layer4 = tf.nn.relu(full_layer4)

    # # 最后的全连接层只有1个输出
    # with tf.variable_scope('full5') as scope:
    #     full_weight5 = truncated_normal_var(name='full_mult5', shape=[64, num_targets], dtype=tf.float32)
    #     full_bias5 = zero_var(name='full_bias5', shape=[num_targets], dtype=tf.float32)
    #     final_output = tf.add(tf.matmul(full_layer4, full_weight5), full_bias5)

    # 最后一层全连接层
    with tf.variable_scope('full') as scope:
        final_output = tf.layers.dense(full_layer4, num_targets, activation=None, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
    return final_output


# 定义评价函数
# 计算MAE平均绝对误差
def MAE(logits, targets):
    mae = tf.reduce_mean(tf.abs(tf.subtract(logits, targets)))  # 平均绝对误差
    return mae


# 计算MAPE 平均绝对百分误差
def MAPE(logits, targets):
    mape = tf.reduce_mean(tf.truediv(tf.abs(tf.subtract(logits, targets)), targets))  # 平均绝对百分比误差
    return mape


# 损失函数MSE
def MSE(logits, targets):
    mse = tf.reduce_mean(tf.square(logits - targets), name='mse')  # 均方误差
    return mse


# 训练阶段函数
def train_step(loss_value, generation_num):
    # 自适应学习率递减
    model_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=generation_num,
                                                     decay_steps=num_gens_to_wait, decay_rate=lr_decay, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(learning_rate=model_learning_rate).minimize(loss=loss_value,
                                                                                       global_step=generation_num)
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


# 计算RMSE均方根误差
def RMSE(logits, targets):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, targets))))  # 均方根误差
    return rmse


# 计算RPD值
def RPD(logits, targets):
    mean_of_targets = tf.reduce_mean(targets)
    stdev = tf.sqrt(tf.divide(tf.reduce_sum(tf.square(tf.subtract(targets, mean_of_targets))),
                              (BATCH_SIZE - 1)))  # 测定值标准差
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, targets))))  # 测定值均方误差
    rpd = tf.divide(stdev, rmse)
    return rpd


# 获取数据
print('Now getting and transforming Data')
train_images, train_targets = create_pipeline(filename=TRAIN_FILE, batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                                              num_epochs=NUM_EPOCHS)

test_images, test_targets = create_pipeline(filename=TEST_FILE, batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                                            num_epochs=NUM_EPOCHS)

# 声明模型
print('Creating the CNN model.')
is_train = tf.placeholder(tf.bool)
with tf.variable_scope('model_definition') as scope:
    model_output = inference(input_images=train_images, batch_size=BATCH_SIZE, is_training=is_train)
    # 这非常重要，我们必须设置scope重用变量
    # 否则，当我们设置测试网络模型，它会设置新的随机变量，这会使在测试批次上进行随机评估，影响评估结果
    scope.reuse_variables()
    test_output = inference(input_images=test_images, batch_size=BATCH_SIZE, is_training=is_train)
# 声明损失函数
print('Declare Loss Function.')
loss = RMSE(logits=model_output, targets=train_targets)
loss_in_testdata = RMSE(logits=test_output, targets=test_targets)

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
train_mae_tl = []
train_mape_tl = []
train_lossl = []
train_r2_tl = []
train_mse_tl = []

test_mae_tl = []
test_mape_tl = []
test_lossl = []
test_r2_tl = []
test_mse_tl = []
test_rpd_tl = []

saver = tf.train.Saver(max_to_keep=20)
for i in tqdm.tqdm(range(generations)):
    """ _, loss_value = sess.run([train_op, loss], {is_training: True})
    # 根据Udacity中Batch normalization的教程，除了在梯度下降的时候使用is_training:True其余时候均将其设置为False
    # 显示在训练集上各项指标
    # 因为在前100次迭代过程中会有明显的下降过程不能分清细节"""
    # 因为我们使用了BatchNormalization算法，所以我们训练和输出前向传播结果的过程要分开
    sess.run(train_op, {is_train: True})
    if i >= INVALID_NUMBER:
        if (i + 1)%output_every == 0:
            Train_Targets, Model_Output, Test_Targets, Test_Output = sess.run(
                [train_targets, model_output, test_targets, test_output], {is_train: False})
            # 因为这个函数中使用了tensorflow所定义的矩阵运算的方法，所以此处一定要用Sess.run的方法来计算
            train_Mae = MAE(Model_Output, Train_Targets)
            train_Mape = MAPE(Model_Output, Train_Targets)
            train_Mse = MSE(Model_Output, Train_Targets)
            train_Rmse = RMSE(Model_Output, Train_Targets)
            train_R2 = R2(Model_Output, Train_Targets)
            train_mae, train_mape, train_mse, train_rmse, train_r2 = sess.run(
                [train_Mae, train_Mape, train_Mse, train_Rmse, train_R2])

            test_Mae = MAE(Test_Output, Test_Targets)
            test_Mape = MAPE(Test_Output, Test_Targets)
            test_Mse = MSE(Test_Output, Test_Targets)
            test_Rmse = RMSE(Test_Output, Test_Targets)
            test_R2 = R2(Test_Output, Test_Targets)
            test_Rpd = RPD(Test_Output, Test_Targets)
            test_mae, test_mape, test_mse, test_rmse, test_r2, test_rpd = sess.run(
                [test_Mae, test_Mape, test_Mse, test_Rmse, test_R2, test_Rpd])

            if test_mae < MAX_Test_MAE:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
            elif test_mape < MAX_Test_MAPE:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
            elif test_rmse < MAX_Test_RMSE:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
            elif test_mse < MAX_Test_MSE:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
            elif test_r2 > MIN_Test_R:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
            elif test_rpd > MIN_Test_RPD:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)

            if (i + 1)%ViewDate == 0:
                print('训练集批次数据的前10个数据\n', Train_Targets[:10])
                print('训练集批次数据通过inference的前10个输出结果\n', Model_Output[:10])
                print('测试集批次数据的前10个数据\n', Test_Targets[:10])
                print('测试集批次数据中通过inference后的前10个输出结果\n', Test_Output[:10])

            train_mae_tl.append([train_mae])
            train_mape_tl.append([train_mape])
            train_lossl.append([train_rmse])
            output = 'Generation {}: train Loss = {:.5f}'.format((i + 1), train_rmse)
            train_r2_tl.append([train_r2])
            train_mse_tl.append([train_mse])
            print(output)  # 显示训练集上的loss值

            # 显示在测试集上各项指标
            test_mae_tl.append([test_mae])
            test_mape_tl.append([test_mape])
            test_lossl.append([test_rmse])
            test_loss_output = 'Generation {}: test Loss = {:.5f}'.format((i + 1), test_rmse)
            test_r2_tl.append([test_r2])
            test_mse_tl.append([test_mse])
            test_rpd_tl.append([test_rpd])
            print(test_loss_output)
        # 保存所有属性值
        if (i + 1)%SAVEValue == 0:
            save_Totest_file = str(i + 1) + save_test_file
            with open(save_Totest_file, "w", newline='') as f:
                writer = csv.writer(f)
                for a in range(test_mae_tl.__len__()):
                    writer.writerows([test_mae_tl[a]])
                for a in range(test_mape_tl.__len__()):
                    writer.writerows([test_mape_tl[a]])
                for a in range(test_lossl.__len__()):
                    writer.writerows([test_lossl[a]])
                for a in range(test_r2_tl.__len__()):
                    writer.writerows([test_r2_tl[a]])
                for a in range(test_mse_tl.__len__()):
                    writer.writerows([test_mse_tl[a]])
                for a in range(test_rpd_tl.__len__()):
                    writer.writerows([test_rpd_tl[a]])
            f.close()

            save_Totrain_file = str(i + 1) + save_train_file
            with open(save_Totrain_file, "w", newline='') as f:
                writer = csv.writer(f)
                for a in range(train_mae_tl.__len__()):
                    writer.writerows([train_mae_tl[a]])
                for a in range(train_mape_tl.__len__()):
                    writer.writerows([train_mape_tl[a]])
                for a in range(train_lossl.__len__()):
                    writer.writerows([train_lossl[a]])
                for a in range(train_r2_tl.__len__()):
                    writer.writerows([train_r2_tl[a]])
                for a in range(train_mse_tl.__len__()):
                    writer.writerows([train_mse_tl[a]])
            f.close()

        if (i + 1)%ViewGraph == 0:
            # 打印损失函数
            output_indices = range(INVALID_NUMBER, i + 1, output_every)
            eval_indices = range(INVALID_NUMBER, i + 1, output_every)

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

            # 显示训练集/测试集MAE函数变化
            plt.plot(output_indices, train_mae_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_mae_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' MAE of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('MAE')
            plt.legend(loc=1, fancybox=True, shadow=True)
            plt.savefig('MAE_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示训练集/测试集MAPE函数变化
            plt.plot(output_indices, train_mape_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_mape_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' MAPE of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('MAPE')
            plt.legend(loc=1, fancybox=True, shadow=True)
            plt.savefig('MAPE_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示训练集/测试集R2函数变化
            plt.plot(output_indices, train_r2_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_r2_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' R of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('R')
            plt.legend(loc=4, fancybox=True, shadow=True)
            plt.savefig('R_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示训练集/测试集MSE函数变化
            plt.plot(output_indices, train_mse_tl, label='train dataset', linewidth=1.0, color='red', linestyle='--')
            plt.plot(eval_indices, test_mse_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' MSE of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('MSE')
            plt.legend(loc=1, fancybox=True, shadow=True)
            plt.savefig('MSE_' + str(i + 1) + '.png', dpi=300)
            plt.show()

            # 显示测试集RPD函数变化
            plt.plot(eval_indices, test_rpd_tl, label='test dataset', linewidth=1.0, color='blue')
            plt.title(' RPD of Tensile')
            plt.xlabel('Generation')
            plt.ylabel('RPD')
            plt.legend(loc=2, fancybox=True, shadow=True)
            plt.savefig('RPD_' + str(i + 1) + '.png', dpi=300)
            plt.show()
            plt.close()

# 关闭线程和Session
coord.request_stop()
coord.join(threads)
sess.close()
