import tensorflow as tf
import cifar10_input
import os
import cifar10_download
import numpy as np
from config import FLAGS


data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

images_batch, label_batch = cifar10_input.train_inputs(data_dir=data_dir, batch_size=100)

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner，此时文件名队列已经进队

    images, label = sess.run([images_batch, label_batch])
    print(images)
    print(np.shape(images))

    coord.request_stop()
    coord.join(threads)
