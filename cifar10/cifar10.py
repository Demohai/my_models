import re
import cifar10_input
from config import FLAGS

import tensorflow as tf

import cifar10_download

# Global constants describing the cifar-10 data set
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = cifar10_download.DATA_URL


def _activation_summary(x):
    """Helper to create summaries for activation

    Creates a summary that provides a histogram of activations
    Creares a summary that measures the sparsity of activations

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_initializer(name, shape, initializer):
    """Helper to create a variable stored on CPU memory

    Args:
        name: name of the variables
        shape: lists of ints
        initializer: a type of initializer

    Returns:
        variable tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def inference(images):
    """Build the cifar-10 model

    Args:
        images: images input

    Returns:
        Logits
    """
    # We instantiate all variables using tf.get_Variable( ) instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all variables of tf.get_Variable() with tf.Variable()

    # conv1
    with tf.variable_scope('conv1'):
        kernel = _variable_initializer('weights',
                                       shape=[5, 5, 3, 64],
                                       initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_initializer('biases',
                                       shape=[64],
                                       initializer=tf.constant_initializer(0.0))
        pre_activation = conv + biases
        conv1 = tf.nn.relu(pre_activation, name='conv1')
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2'):
        kernel = _variable_initializer('weights',
                                       shape=[5, 5, 64, 64],
                                       initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_initializer('biases',
                                       shape=[64],
                                       initializer=tf.constant_initializer(0.1))
        pre_activation = conv + biases
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # norm2
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # fc1
    with tf.variable_scope('fc1'):
        # flatten the norm2
        reshape = tf.reshape(norm2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value

        weights = _variable_initializer('weights',
                                        shape=[dim, 384],
                                        initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = _variable_initializer('biases',
                                       shape=[384],
                                       initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='fc1')
        _activation_summary(fc1)

    # fc2
    with tf.variable_scope('fc2'):
        weights = _variable_initializer('weights',
                                        shape=[384, 192],
                                        initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        biases = _variable_initializer('biases',
                                       shape=[192],
                                       initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name='fc2')
        _activation_summary(fc2)

    # fc3
    with tf.variable_scope('fc3'):
        weights = _variable_initializer('weights',
                                        shape=[192, NUM_CLASSES],
                                        initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32))
        biases = _variable_initializer('biases',
                                       shape=[NUM_CLASSES],
                                       initializer=tf.constant_initializer(0.0))
        fc3 = tf.add(tf.matmul(fc2, weights), biases, name='fc3')
        _activation_summary(fc3)

    return fc3


def loss(logits, labels):
    """ Not add L2 loss to all the trainable variables with respect to the original part

    Args:
        logits: returned from inference()
        labels: the labels of images

    Returns:
        Loss tensor of type float
    """
    # Calculate the average cross entropy loss across the batch
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# def _add_loss_summaries(cross_entropy_mean):
#     """ Add summaries for losses in CIFAR-10 model
#
#         Generate moving average for all losses and associated summaries for
#         visualizing the performance of the network.
#
#     Args:
#         cross_entropy_mean: returned from loss()
#
#     Returns:
#         loss_average_op: op for generating moving averages of losses.
#     """
#     # Compute the moving average of all individual losses and the total loss
#     loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
def train(cross_entropy_mean, global_step):
    """ Train the CIFAR-10 model

    Args:
        cross_entropy_mean: returned from loss()
        global_step: the step of training

    Returns:
        train_op: the operation of training
    """
    # num_batches_per_epoch = 50000 / 128 = 390
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # decay_steps = 390 * 350 = 97500
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean, global_step=global_step)
    return train_op
