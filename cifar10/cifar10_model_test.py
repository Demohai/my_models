import tensorflow as tf
import cifar10_input
import cifar10
from config import FLAGS
import os
import numpy as np
import cifar10_download
from datetime import datetime


def evaluate():
    # Get test images and labels for the cifar10
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    eval_images_batch, eval_labels_batch = cifar10_input.eval_inputs(data_dir, FLAGS.batch_size)

    # Compute the logits predictions with the input of test_images
    logits = cifar10.inference(eval_images_batch)

    # Calculate the predictions
    # top_k_op is a tensor of bool
    top_k_op = tf.nn.in_top_k(logits, eval_labels_batch, 1)

    # Calculate the number of iterations to run through test data set once
    num_iters = int(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size)
    total_sample_count = num_iters * FLAGS.batch_size

    # Counts the number of correct predictions
    true_count = 0
    step = 0
    # Create an object of class Saver() named saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Returns CheckpointState proto from the "checkpoint" file which is in the checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores model from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")

        # Test whether the model restored using the variable of conv1/weights
        # graph = tf.get_default_graph()
        # w1 = graph.get_tensor_by_name("conv1/weights:0")
        # print(sess.run(w1))
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner，此时文件名队列已经进队

        while step < num_iters:
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

        print("true_count = ", true_count)
        print("step = ", step)
        # Compute precision
        precision = true_count / total_sample_count
        print("%s: precision = %.3f" % (datetime.now(), precision))
        coord.request_stop()
        coord.join(threads)


def main(_):
    # Download the data set
    cifar10_download.download_data()
    #  Determines whether a eval_dir exists or not.
    if tf.gfile.Exists(FLAGS.eval_dir):
        # Deletes everything under eval_dir recursively.
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    # Creates a directory and all parent/intermediate directories.
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == "__main__":
    tf.app.run()