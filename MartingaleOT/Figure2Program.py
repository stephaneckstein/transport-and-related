#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import os


# Sampling from reference measure $\mu$
def gen_points(batch_size):
    while True:
        dataset = np.random.random_sample([batch_size, 2])
        yield(dataset*[2, 4] - [1, 2])


# General layer structure to approximate the univariate functions h_1,...,h_d
def univ_approx(x, name, hidden_dim=64):
    with tf.variable_scope(name):
        ua_w = tf.get_variable('ua_w', shape=[1, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z)
        ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z2 = tf.matmul(a, ua_w2) + ua_b2
        a2 = tf.nn.relu(z2)
        ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z3 = tf.matmul(a2, ua_w3) + ua_b3
        a3 = tf.nn.relu(z3)
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return z


def build_graph(penalty_method='L2', rho=2.3, batch_size=2**10):
    """

    :param penalty_method: Only L^2 penalty function ('L2') currently included
    :param rho: parameter of function f
    :param batch_size: needed here (compared to other programs) since our implementation of the function f needs it
    :return:
    """

    d = 2  # Dimension - At the moment this is fix here as no higher dimensions were needed.
    # Declare function f with parameter rho

    def f(p):
        return tf.reshape(-tf.pow(tf.abs(p[:, 1] - p[:, 0]), rho), [batch_size, 1])

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, d])  # Random samples
    gamma = tf.placeholder(dtype=tf.float32, shape=[1])  # penalty factor
    s = 0
    for i in range(d):
        s += univ_approx(x[:, i:i+1], str(i))
    Ints = tf.reduce_mean(s)
    h3 = univ_approx(x[:, 0:1], 'Mart')
    diff_t = h3 * (x[:, 1:2] - x[:, 0:1])

    if penalty_method == 'L2':
        target_function = Ints + gamma * tf.reduce_mean(tf.square(tf.nn.relu(f(x)-s-diff_t)))
    else:
        raise ValueError('Your chosen penalty method is not implemented')
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(target_function)
    return x, gamma, train_op, target_function


def run_experiments(x, gamma, train_op, target_function, num_experiments=9, num_runs_each=100,
                    gamma_values=[10*(2**k) for k in range(2, 2+9)], batch_values=[2 ** 10]*9, n=20000, final_n=1000):
    """

    :param x: Input sample placeholder in tensorflow graph
    :param gamma: penalisation factor placeholder in tensorflow graph
    :param train_op: optimizer in tensorflow graph
    :param target_function: objective function in tensorflow graph
    :param num_experiments: Each experiment runs the same batch_size and gamma value for num_runs_each many times.
    :param num_runs_each: see num_experiments
    :param gamma_values: list of gamma values for different experiments
    :param batch_values: list of batch size values for different experiments
    :param n: number of iterations in the training
    :param final_n: final number of iterations that is averaged to obtain the value of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :return: outputs and saves final values for each individual experiment
    """
    # config to run program on a CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0})
    for i in range(num_experiments):
        value_list = []
        start_time = time.time()
        gamma_value = gamma_values[i]
        batch_size = batch_values[i]
        for j in range(num_runs_each):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                gen_ = gen_points(batch_size)
                final_value = 0
                for t in range(n):
                    input_sample = next(gen_)
                    _ = sess.run([train_op], feed_dict={x: input_sample, gamma: [gamma_value]})
                for t in range(final_n):
                    input_sample = next(gen_)
                    (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value]})
                    final_value += c[0]

                final_value /= final_n
                print('Run ' + str(j) + ' has final value = ' + str(final_value))
                value_list.append(final_value)

        print('Gamma value = ' + str(gamma_value))
        print(np.mean(value_list))
        print(np.percentile(value_list, 97.5))
        print(np.percentile(value_list, 2.5))
        print(np.std(value_list))
        print(time.time()-start_time)
        directory = os.path.dirname(__file__)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt(directory+'/Gamma_'+str(gamma_value)+time_string+'.txt', value_list)

if __name__ == '__main__':
    # All arguments defined in the functions and standard parameters as in the paper.
    # Note that the main difference to other programs is that here, the batch size has to be fix for all experiments
    # and already needs to be specified when building the graph.
    # Further, currently the graph only allows dimension 2 but this could be adjusted easily

    n_experiments = 3
    n_runs_each = 100
    gamma_values = [10 * (2 ** k) for k in range(8, 8 + n_experiments)]
    batch_values = [2 ** 10] * n_experiments
    n = 20000
    final_n = 1000

    X, gamma_1, train_op_1, target_function_1 = build_graph()
    run_experiments(x=X, gamma=gamma_1, target_function=target_function_1, train_op=train_op_1,
                    num_experiments=n_experiments, num_runs_each=n_runs_each, gamma_values=gamma_values,
                    batch_values=batch_values, n=n, final_n=final_n)

