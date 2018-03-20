import tensorflow as tf
import numpy as np

BATCH_SIZE = 4


def RMSE(logits, targets):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, targets))))  # 均方根误差
    return rmse


# 计算RPD值
def RPD(logits, targets):
    mean_of_logits = tf.reduce_mean(logits)
    stdev = tf.sqrt(
        tf.divide(tf.reduce_sum(tf.square(tf.subtract(logits, mean_of_logits))),
                  tf.cast((BATCH_SIZE - 1), tf.double)))  # 测定值标准差
    rmse = RMSE(logits, targets)  # 测定值均方误差
    rpd = tf.divide(stdev, rmse)
    return logits, targets, mean_of_logits, stdev, rmse, rpd


def RPD_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile, logits_Yeild, logits_Elongation = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile, targets_Yeild, targets_Elongation = tf.unstack(tf.cast(trans_targets, tf.double))
    l_1, t_1, m_1, s_1, rmse_1, rpd_1 = RPD(logits_Tensile, targets_Tensile)
    l_2, t_2, m_2, s_2, rmse_2, rpd_2 = RPD(logits_Yeild, targets_Yeild)
    l_3, t_3, m_3, s_3, rmse_3, rpd_3 = RPD(logits_Elongation, targets_Elongation)
    return l_3, t_3, m_3, s_3, rmse_3, rpd_3


a = tf.constant(np.array([[2, 3, 4], [3, 4, 5], [6, 7, 8], [3, 6, 5]]))
b = tf.constant(np.array([[1, 3, 4], [2, 5, 6], [3, 4, 5], [7, 8, 10]]))
sess = tf.Session()
l, t, m, s, rmse, rpd = sess.run(RPD_of_batch(a, b))

print('l', l)
print('t', t)
print('m', m)
print('s', s)
print('rmse', rmse)
print('rpd', rpd)

# l [ 2.  3.  6.  3.]
# t [ 1.  2.  3.  7.]
# m 3.5
# s 1.73205080757
# rmse 2.59807621135
# rpd 0.666666666667

# pow((2.0 - 3.5), 2.0) + pow((3.0 - 3.5), 2.0) + pow((6.0 - 3.5), 2.0) + pow((3.0 - 3.5), 2.0)
# (2.25+0.25+6.25+0.25)/3=3
# pow(3,0.5)=1.73205080757