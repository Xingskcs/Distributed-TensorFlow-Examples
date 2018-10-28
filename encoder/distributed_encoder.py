""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/Xingskcs/Distributed-TensorFlow-Examples
"""
from __future__ import division, print_function, absolute_import

import time

import tensorflow as tf
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for definig the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("data_dir", "/data_dir",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_string("model_dir", "/tmp/checkpoints",
                     "Directory for storing the checkpoints")
tf.app.flags.DEFINE_integer("workers", 3, "Number of workers")
tf.app.flags.DEFINE_integer("ps", 1, "Number of ps")

FLAGS = tf.app.flags.FLAGS

# Global variables
learning_rate = 0.01
num_steps = 100000
batch_size = 256

num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)


def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""

    with tf.device("/job:ps/task:%d" % (i)):
        return tf.FIFOQueue(FLAGS.workers, tf.int32, shared_name="done_queue" + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(len(FLAGS.ps_hosts.split(",")))]


# Building the encoder
def encoder(x, weights, biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x, weights, biases):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    
    if FLAGS.job_name == "ps":
        sess = tf.Session(server.target)
        queue = create_done_queue(FLAGS.task_index)

        # wait until all workers are done
        for i in range(FLAGS.workers):
            sess.run(queue.dequeue())
            print("ps %d received done %d" % (FLAGS.task_index, i))

        print("ps %d: quitting" % (FLAGS.task_index))
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            weights = {
                'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
                'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
                'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
                'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
                'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'decoder_b2': tf.Variable(tf.random_normal([num_input])),
            }
            X = tf.placeholder("float", [None, num_input])
            # Construct model
            encoder_op = encoder(X, weights, biases)
            decoder_op = decoder(encoder_op, weights, biases)
            # Prediction
            y_pred = decoder_op
            # Targets (Labels) are the input data.
            y_true = X
            # Define loss and optimizer, minimize the squared error.
            loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            global_step = tf.Variable(0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()

            enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                enq_ops.append(qop)
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.model_dir,
                                 init_op=init,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        begin_time = time.time()
        config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index])
        with sv.managed_session(server.target, config=config) as sess:
            step = 0
            local_step = 0
            sess.run(init)
            while not sv.should_stop() and step < num_steps:
                start_time = time.time()
                # Get the next batch of MNIST data (only images are needed, not labels)
                batch_x, _ = mnist.train.next_batch(batch_size)
                _, l, step = sess.run([optimizer, loss, global_step], feed_dict={X: batch_x})
                elapsed_time = time.time() - start_time
                print ("Global step %d" % step, "Local step %d" % local_step, " AvgTime: %3.2fms" % float(elapsed_time * 1000))
                local_step += 1

            print("Total Time: %3.2fs" % float(time.time() - begin_time))

            # Signal to ps shards that we are done
            for op in enq_ops:
                sess.run(op)
        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()



