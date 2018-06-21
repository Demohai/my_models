"""
This file includes PrimaryCaps and DigitCaps, with dynamic routing and squashing defined here
author: Zhu Hai
address:

"""

import tensorflow as tf
import numpy as np
from config import cfg

epsilon = 1e-9

class CapsLayer(object):
    """
    PrimaryCaps and DigitCaps
    Args:
        input: a 4-D tensor, the output of normal convolution
        num_outputs: the number of outputs of PrimaryCaps
        vec_len: integer, the length of output vectors of PrimaryCaps
        layer_type: string, PrimaryCaps or DigitCaps

    Returns:
        PrimaryCaps: a 4-D tensor
        DigitCaps: a 2-D tensor
    """
    def __int__(self, num_outputs_p, num_outputs_d, vec_len_p, vec_len_d):
        self.num_outputs_p = num_outputs_p    # the number of PrimaryCaps outputs
        self.num_outputs_d = num_outputs_d    # the number of DigitCaps outputs
        self.vec_len_p = vec_len_p            # the length of a vector of PrimaryCaps
        self.vec_len_d = vec_len_d            # the length of a vector of DigitCaps

    def PrimaryCaps(self, input, kernel_size, stride):
        # kernel_size and stride parameters are used, if the layer_type = PrimaryCaps
        # PrimaryCaps, a convolution layer
        # input: [batch_size, 20, 20, 256]
        self.kernel_size = kernel_size
        self.stride = stride
        capsules = tf.layers.conv2d(input, self.num_outputs_p * self.vec_len_p,
                                            self.kernel_size, self.stride, padding="VALID",
                                            activation=tf.nn.relu)
        capsules_shape = capsules.get_shape().as_list()
        # shape: (batch_size, 1152, 8, 1)
        capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len_p, 1))
        # squashing
        capsules = squash(capsules)

        return capsules

    def DigitCaps(self, input):
        # DigitCaps
        # input: the output of PrimaryCaps; shape: (batch_size, 1152, 8, 1)
        # reshape input into (batch_size, 1152, 1, 8, 1)
        input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

        with tf.variable_scope('routing'):
            # b_ij shape: (batch_size, 1152, 1, 8, 1)
            b_ij = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs_d, 1, 1],
                                        dtype=tf.float32))
            capsules = routing(input, b_ij, self.num_outputs_d, self.vec_len_d, self.vec_len_p)
            capsules = tf.squeeze(capsules, axis=1)

        return capsules

# routing algorithm
def routing(input, b_ij, num_outputs_d, vec_len_d, vec_len_p):
    """
    Args:
        input: (batch_size, 1152, 1, 8, 1)
        b_ij:  (batch_size, 1152, 10, 1, 1)
        num_outputs_d: integer, the number of DigitCaps outputs
        vec_len_d: integer, the length of a vector of DigitCaps
        vec_len_p: integer, the length of a vector of PrimaryCaps
    Returns:
         a tensor of shape (batch_size, 1, 10, 16, 1)
    """
    W = tf.get_variable('Weight', shape=(1, input.shape[1].value, num_outputs_d, vec_len_p, vec_len_d),
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # input shape: (batch_size, 1152, 1, 8, 1) => (batch_size, 1152, 10, 8, 1)
    # W shape: (1, 1152, 10, 8, 16) => (batch_size, 1152, 10, 8, 16)
    input = tf.tile(input, [1, 1, num_outputs_d, 1, 1])
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])

    # in last 2 dims:
    # (8, 16).T * (8, 1) => (16, 1) => (batch_size, 1152, 10, 16, 1)
    u_hat = tf.matmul(W, input, transpose_a=True)

    # routing for 3 iterations
    for iter in range(cfg.iter_routing):
        with tf.variable_scope('iter' + str(iter)):
            # c_ij shape: (batch_size, 1152, 10, 1, 1)
            c_ij = tf.nn.softmax(b_ij, dim=2)
            # s_j shape: (batch_size, 1152, 10, 16, 1)
            s_j = tf.multiply(c_ij, u_hat)
            # sum in the second dim, resulting in (batch_size, 1, 10, 16, 1)
            s_j = tf.reduce_sum(s_j, axis=1, keepdims=True)
            # v_j shape: (batch_size, 1, 10, 16, 1)
            v_j = squash(s_j)
            # v_j_tiled shape: (batch_size, 1152, 10, 16, 1)
            v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])
            b_ij += tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            # the last iteration, return v_j
            if iter == cfg.iter_routing - 1:
                return v_j

# squashing
def squash(vector):
    """
    Args:
        vector: a tensor with shape (batch_size, 1, 10, 16, 1)
    Returns:
        a tensor with the same shape, but each entry is between 0 and 1
    """
    vec_squared_norm = tf.reduce_sum(tf.square(vector), 3, keepdims=True)
    # vec_squared_norm + epsilon to avoid vec_squared_norm equals 0
    vec_squashed = (vec_squared_norm / (1 + vec_squared_norm)) * (vector / tf.sqrt(vec_squared_norm + epsilon))
    return vec_squashed
