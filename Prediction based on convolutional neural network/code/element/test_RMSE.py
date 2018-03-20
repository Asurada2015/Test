import tensorflow as tf
import numpy as np


def RMSE(logits, targets):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(logits - targets)))  # 均方根误差
    return (targets, logits, rmse)


def RMSE_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile, logits_Yeild, logits_Elongation = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile, targets_Yeild, targets_Elongation = tf.unstack(tf.cast(trans_targets, tf.double))
    t_1, l_1, rmse_1 = RMSE(logits_Tensile, targets_Tensile)
    t_2, l_2, rmse_2 = RMSE(logits_Yeild, targets_Yeild)
    t_3, l_3, rmse_3 = RMSE(logits_Elongation, targets_Elongation)
    return t_1, l_1, rmse_1


a = tf.constant(np.array([[2, 3, 4], [3, 4, 5], [6, 7, 8], [3, 6, 5]]))
b = tf.constant(np.array([[1, 3, 4], [2, 5, 6], [3, 4, 5], [7, 8, 10]]))
sess = tf.Session()
t, l, r = sess.run(RMSE_of_batch(a, b))

print('t', t)   #  t [ 1.  2.  3.  7.]
print('l', l)   #  l [ 2.  3.  6.  3.]
print('r', r)   #  r 2.59807621135
