# 使用Tensorflow读出csv文件中的数据。
import tensorflow as tf

MIN_AFTER_DEQUEUE = 1000
BATCH_SIZE = 100
NUM_EPOCHS = 1000
NUM_THREADS = 2


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                [0.], [0.], [0.], [0.]]
    C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, H, B, Furnace, RoughMill, FinishMill, DownCoil, Tensile, Yeild, Elongation \
        = tf.decode_csv(value, defaults)
    vertor_example = tf.stack(
        [C, MN, SI, P, S, CU, AL, ALS, NI, CR, TI, MO, V, NB, N, H, B, Furnace, RoughMill, FinishMill,
         DownCoil])
    vertor_label = tf.stack([Tensile, Yeild, Elongation])

    return vertor_example, vertor_label


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


x_train_batch, y_train_batch = create_pipeline('a_train.csv', batch_size=BATCH_SIZE, num_threads=NUM_THREADS,
                                               num_epochs=NUM_EPOCHS)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(local_init_op)
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(4):
        example, label = sess.run([x_train_batch, y_train_batch])
        print(sess.run(tf.shape(example)))
        print(sess.run(tf.shape(label)))
        print(example[1][1])
        # print(label)
    coord.request_stop()
    coord.join(threads)
