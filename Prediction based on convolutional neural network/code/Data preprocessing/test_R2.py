import tensorflow as tf
import numpy as np


def R2(logits, targets):
    mean_targets = tf.reduce_mean(targets)
    # ss_tot总体平方和
    ss_tot = tf.reduce_sum(tf.square(tf.subtract(targets, mean_targets)))
    # ss_err残差平方和
    ss_err = tf.reduce_sum(tf.square(tf.subtract(logits, targets)))
    r2 = 1 - (tf.truediv(ss_err, ss_tot))
    return (targets, logits, mean_targets, ss_tot, ss_err, r2)


def R2_of_batch(logits, targets):
    trans_logits = tf.transpose(logits)
    logits_Tensile, logits_Yeild, logits_Elongation = tf.unstack(tf.cast(trans_logits, tf.double))
    trans_targets = tf.transpose(targets)
    targets_Tensile, targets_Yeild, targets_Elongation = tf.unstack(tf.cast(trans_targets, tf.double))
    t_1, l_1, m_1, ss_t_1, ss_r_1, r2_1 = R2(logits_Tensile, targets_Tensile)
    t_2, l_2, m_2, ss_t_2, ss_r_2, r2_2 = R2(logits_Yeild, targets_Yeild)
    t_3, l_3, m_3, ss_t_3, ss_r_3, r2_3 = R2(logits_Elongation, targets_Elongation)
    return t_3, l_3, m_3, ss_t_3, ss_r_3, r2_3


a = tf.constant(np.array([[2, 3, 4], [3, 4, 5], [6, 7, 8], [3, 6, 5]]))
b = tf.constant(np.array([[1, 3, 4], [2, 5, 6], [3, 4, 5], [7, 8, 10]]))
sess = tf.Session()
t, l, m, ss_t, ss_r, r2 = sess.run(R2_of_batch(a, b))
print('t', t)  # t [ 1.  2.  3.  7.]
print('l', l)  # l [ 2.  3.  6.  3.]
print('m', m)  # m 3.25 (1+2+3+7)/4=3.25
print('ss_t', ss_t)  # pow((3.25-1),2)+pow((2-3.25),2)+pow((3-3.25),2)+pow((7-3.25),2)=20.75
print('ss_r', ss_r)  # pow((1-2),2)+pow((2-3),2)+pow((3-6),2)+pow((7-3),2)=27
print('r2', r2)# (1-27/20.75)=-0.301204819277
