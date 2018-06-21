import tensorflow as tf

from config import cfg
from utils import get_batch_data
from capsLayer import CapsLayer

epsilon = 1e-9

class CapsNet(object):
    def __int__(self, is_training=True):
        if is_training:
            self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
            self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

            self.build_arch()
            self._summary()

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
        else:
            self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
            self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
            self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('conv1_layer'):
            # conv1, return: (batch_size, 20, 20, 256)
            conv1 = tf.layers.conv2d(self.X, filters=256, kernel_size=9, strides=1, padding='VALID')

            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # PrimaryCaps, return (batch_size, 1152, 8, 1)
        with tf.variabel_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs_p=32, num_outputs_d=10, vec_len_p=8, vec_len_d=16)
            caps1 = primaryCaps.PrimaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape == [cfg.batch_size, 1152, 8, 1]

        # DigitCaps, return (batch_size, 10, 16, 1)
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs_p=32, num_outputs_d=10, vec_len_p=8, vec_len_d=16)
            self.caps2 = digitCaps.DigitCaps(caps1)
            assert self.caps2.get_shape == [cfg.batch_size, 10, 16, 1]

        with tf.variable_scope('masking'):
            self.maked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
            self.v_length =
