# 修改read_data函数csv中数据成为一个四维图像矩阵
import tensorflow as tf

BATCH_SIZE = 400
NUM_THREADS = 2
MAX_NUM = 500


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.]]
    NUM, C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, B, Furnace, RoughMill, FinishMill, DownCoil, Tensile, Yeild, Elongation \
        = tf.decode_csv(value, defaults)
    vertor_example = tf.stack(
        [C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, B, Furnace, RoughMill, FinishMill,
         DownCoil])
    # 将(21)维度的数据添加维度成为(1,21)的向量

    example_2D = tf.expand_dims(vertor_example, 0)
    trans_example_2D = tf.transpose(example_2D)
    train_example = tf.expand_dims(tf.matmul(trans_example_2D, example_2D), 2)
    vertor_label = tf.stack([Tensile])
    vertor_num = tf.stack([NUM])

    # return vertor_example, vertor_label
    # (21);(3)使用batch后为(batch_size,21)
    # return example_2D(1,21), trans_example_2D(21,1)
    return train_example, vertor_label, vertor_num


def create_pipeline(filename, batch_size, num_threads):
    file_queue = tf.train.string_input_producer([filename])  # 设置文件名队列
    example, label, no = read_data(file_queue)  # 读取数据和标签

    example_batch, label_batch, no_batch = tf.train.batch(
        [example, label, no], batch_size=batch_size, num_threads=num_threads, capacity=MAX_NUM)

    return example_batch, label_batch, no_batch


x_train_batch, y_train_batch, no_train_batch = create_pipeline('235b_eval_1.csv', batch_size=BATCH_SIZE,
                                                               num_threads=NUM_THREADS)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(local_init_op)
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    example, label, num = sess.run([x_train_batch, y_train_batch, no_train_batch])
    print(sess.run(tf.shape(example)))
    print(sess.run(tf.shape(label)))
    print(sess.run(tf.shape(num)))
    # print(example[0])
    # print(label[0])
    print(label)
    print(num)
    coord.request_stop()
    coord.join(threads)
