""" Generative Adversarial Networks (GAN).

Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.

References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).

Author: csskxing
Project: https://github.com/Xingskcs/Distributed-TensorFlow-Examples
"""

from __future__ import division, print_function, absolute_import

import time

import numpy as np
import tensorflow as tf

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
num_steps = 100000
batch_size = 128
learning_rate = 0.0002

image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points


def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""

    with tf.device("/job:ps/task:%d" % (i)):
        return tf.FIFOQueue(FLAGS.workers, tf.int32, shared_name="done_queue" + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(len(FLAGS.ps_hosts.split(",")))]


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


def generator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def discriminator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


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
                'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
                'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
                'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
                'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
            }
            biases = {
                'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
                'gen_out': tf.Variable(tf.zeros([image_dim])),
                'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
                'disc_out': tf.Variable(tf.zeros([1])),
            }
            gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
            disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')
            # Build Generator Network
            gen_sample = generator(gen_input, weights, biases)
            # Build 2 Discriminator Networks (one from noise input, one from generated samples)
            disc_real = discriminator(disc_input, weights, biases)
            disc_fake = discriminator(gen_sample, weights, biases)
            # Build Loss
            gen_loss = -tf.reduce_mean(tf.log(disc_fake))
            disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
            global_step = tf.Variable(0)
            # Build Optimizers
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # Training variables for each optimizer
            # By Default in Tensorflow, all variables are updated by each optimizer, so we
            # need to precise for each of them the specific variables to update.
            # Generator Network Variables
            gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                        biases['gen_hidden1'], biases['gen_out']]
            # Discriminator Network Variables
            disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                        biases['disc_hidden1'], biases['disc_out']]
            # Create training operations
            train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars, global_step = global_step)
            train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars, global_step = global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            # Initialize the variables(i.e. assign the default values)
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
            sess.run(init)
            step = 0
            local_step = 0
            while not sv.should_stop() and step < num_steps:
                start_time = time.time()
                # Get the next batch of MNIST data (only images are needed, not labels)
                batch_x, _ = mnist.train.next_batch(batch_size)
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

                # Train
                feed_dict = {disc_input: batch_x, gen_input: z}
                _, _, gl, dl, step = sess.run([train_gen, train_disc, gen_loss, disc_loss, global_step],
                                feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                print ("Global step %d" % step, "Local step %d" % local_step, " AvgTime: %3.2fms" % float(elapsed_time * 1000))
                if step % 1000 == 0 or step == 1:
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (step, gl, dl))
                local_step += 1
            print("Total Time: %3.2fs" % float(time.time() - begin_time))

            # Signal to ps shards that we are done
            for op in enq_ops:
                sess.run(op)
        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()