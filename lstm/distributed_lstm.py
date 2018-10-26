import time

import tensorflow as tf
from tensorflow.contrib import rnn

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
learning_rate = 0.001
training_steps = 100000
batch_size = 128

num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)


def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""

    with tf.device("/job:ps/task:%d" % (i)):
        return tf.FIFOQueue(FLAGS.workers, tf.int32, shared_name="done_queue" + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(len(FLAGS.ps_hosts.split(",")))]


def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


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
            # Variables of the hidden layer
            weights = {
                'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([num_classes]))
            }
            X = tf.placeholder("float", [None, timesteps, num_input])
            Y = tf.placeholder("float", [None, num_classes])
            logits = RNN(X, weights, biases)
            prediction = tf.nn.softmax(logits)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=Y))
            global_step = tf.Variable(0)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op, global_step=global_step)
            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
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
            # Loop until the supervisor shuts down or max_steps have complted.
            step = 0
            local_step = 0
            while not sv.should_stop() and step < training_steps:
                start_time = time.time()
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                _, step = sess.run([train_op, global_step], feed_dict={X: batch_x, Y: batch_y})
                elapsed_time = time.time() - start_time
                print ("Global step %d" % step, "Local step %d" % local_step, " AvgTime: %3.2fms" % float(elapsed_time * 1000))
                local_step += 1

            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            # Test trained model, calculate accuracy for 128 mnist test images.
            test_len = 128
            test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            test_label = mnist.test.labels[:test_len]
            print("Test-Accuracy: %2.4f" % sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

            # Signal to ps shards that we are done
            for op in enq_ops:
                sess.run(op)
        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()