import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

ipt = 21  # 输入个数
opt = 3  # 输出个数
hidnum = opt*10  # 隐层结点数量
N = 11  # 取训练数据条数   35443 训练总数据
M = 20  # 取测试数据条数   21473 测试总数据
SUMMARY_DIR = "./log"  # 图记录保存地址
TRAIN_STEPS = 1000
"""定义添加层函数"""

# 此Tensorboard只适用于tensorflow-1.2.1版本的，对于1.4版本并不能显示分布图的结果
# tf.summary.histogram生成直方图
# tf.summary.scalars生成变量图
# 生成变量监控信息并定义生成监控信息日志的操作。

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope(layer_name):
        with tf.variable_scope('weights'):
            Weights = tf.get_variable(name=layer_name + '/weights', shape=[in_size, out_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            variable_summaries(Weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram(layer_name + '/Wx_plus_b', Wx_plus_b)
        with tf.name_scope('out_puts'):
            bn = batch_norm(Wx_plus_b, is_training=True, center=True, scale=True, activation_fn=tf.nn.relu,
                            scope=None)
            outputs = activation_function(bn)
            # 记录神经网络节点输出在经过激活函数之后的分布。
            # outputs = tf.nn.dropout(outputs, 0.8)  # Dropout  可以注释掉 不用
            tf.summary.histogram(layer_name + '/outputs', outputs)
            return outputs


def main():
    TD = pd.read_csv('DT.Train.csv')  # 读取训练数据
    VD = pd.read_csv('DT.Value.csv')  # 读取验证数据

    TDS = TD.head(N)
    VDS = VD.head(M)
    trainx = TDS[list(range(ipt))]  # 取前 21 列
    trainy = TDS[list(range(ipt, ipt + opt))]  # 取后 3 列

    valuex = VDS[list(range(ipt))]  # 取前 21 列
    valuey = VDS[list(range(ipt, ipt + opt))]  # 取后 3 列

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, ipt])
        y_ = tf.placeholder(tf.float32, [None, opt])
        keep_prob = tf.placeholder(tf.float32)  # 表示在dropout后需要保存的数据，keep_prob=1即为所有数据都会得到保存

    l1 = add_layer(x, ipt, hidnum, layer_name='layer1', activation_function=tf.nn.relu)
    l2 = add_layer(l1, hidnum, hidnum, layer_name='layer2', activation_function=tf.nn.relu)
    l3 = add_layer(l2, hidnum, hidnum, layer_name='layer3', activation_function=tf.nn.relu)
    l4 = add_layer(l3, hidnum, hidnum, layer_name='layer4', activation_function=tf.nn.relu)
    l5 = add_layer(l4, hidnum, hidnum, layer_name='layer5', activation_function=tf.nn.relu)
    l6 = add_layer(l5, hidnum, hidnum, layer_name='layer6', activation_function=tf.nn.relu)
    l7 = add_layer(l6, hidnum, hidnum, layer_name='layer7', activation_function=tf.nn.relu)
    l8 = add_layer(l7, hidnum, hidnum, layer_name='layer8', activation_function=tf.nn.relu)
    l9 = add_layer(l8, hidnum, hidnum, layer_name='layer9', activation_function=tf.nn.relu)
    l10 = add_layer(l9, hidnum, hidnum, layer_name='layer10', activation_function=tf.nn.relu)

    with tf.variable_scope('rmse'):
        ww = tf.get_variable("ww", shape=[hidnum, opt], initializer=tf.contrib.layers.xavier_initializer())

        bb = tf.Variable(tf.zeros([opt]))
        ll = tf.matmul(l10, ww) + bb
        rmse = tf.sqrt(tf.reduce_mean(tf.square(ll - y_)))
        tf.summary.scalar('rmse', rmse)

    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.01
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(rmse, global_step=global_step)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            summary, _ = sess.run([merged, optimizer], feed_dict={x: trainx, y_: trainy, keep_prob: 1})
            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的运行信息。
            summary_writer.add_summary(summary, i)
            print('tMSE', sess.run(rmse, feed_dict={x: trainx, y_: trainy, keep_prob: 1}),
                  ', vMSE', sess.run(rmse, feed_dict={x: valuex, y_: valuey, keep_prob: 1}),
                  ', LR', sess.run(learning_rate))

    summary_writer.close()


if __name__ == '__main__':
    main()

    # tensorboard --logdir=D:\CODE\cloudPy\ceshi_00\log
