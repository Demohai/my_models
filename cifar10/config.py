""" Parameters configuration """
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string('data_dir', '/tmp/cifar10_data/',
                    """Path of the CIFAR-10 data""")
flags.DEFINE_integer('batch_size', 128,
                     """Number of images to process in a batch""")
flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                    """Direction where to write event logs and checkpoints""")

flags.DEFINE_integer('max_steps', 1000,
                     """Number of batches or steps to run""")

flags.DEFINE_boolean('log_device_placement', True,
                     """Whether to log device placement""")

flags.DEFINE_integer('log_frequency', 10,
                     """How often to log results to the console""")

flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                    """Directory where to write evaluation event logs.""")

FLAGS = tf.app.flags.FLAGS
