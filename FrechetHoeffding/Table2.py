#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time
import os


# Sampling from reference measure $\mu$
# Note: We centralize the measure here. (In the paper, uniform on [0,1]^d, here uniform on [-0.5, 0.5]^d)
def gen_points(batch_size):
    while True:
        dataset = np.random.random_sample([batch_size, 2])
        yield(dataset-0.5)


# General layer structure to approximate the uni-variate functions h_1,...,h_d
def univ_approx(x, name, hidden_dim=64):
    with tf.variable_scope(name):
        ua_w = tf.get_variable('ua_w', shape=[1, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z)
        ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z2 = tf.matmul(a, ua_w2) + ua_b2
        a2 = tf.nn.relu(z2)
        ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z3 = tf.matmul(a2, ua_w3) + ua_b3
        a3 = tf.nn.relu(z3)
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


# Since we are only using L^2 penalization for the table, tf.float32 is enough (compared to tf.float64 in Figure1.py)
def build_graph(dimension, penalty_method='L2', z_value=(0, 0.25)):
    """Specify network structure

    :param dimension: Here: Number of marginals (i.e. dimension of the whole space). Each marginal has dimension 1
    :param penalty_method: penalty_method: Specifies which penalty to use: Currently 'L2' or 'exp'
    :param z_value: Parameter for the function f
    :return: Various elements of the graph.
    """

    def f(p):
        """ f(p) = 1 if p <= z_value for all elements of p, and f(p) = 0, else

        :param p: tf variable with size [batch_size, dimension] of type float
        :return: tf variable with size [batch_size, 1] of type tf.float32
        """
        cond1 = tf.greater(p, z_value)
        cond1 = tf.reduce_any(cond1, 1)
        cond1 = tf.logical_not(cond1)
        out = tf.cast(cond1, dtype=tf.float32)
        return out

    x = tf.placeholder(dtype=tf.float32, shape=[None, dimension])  # Samples from $\mu$, also used for marginals
    gamma = tf.placeholder(dtype=tf.float32, shape=[1])  # penalty factor

    s = 0
    for i in range(dimension):
        s += univ_approx(x[:, i:i+1], str(i))
    ints = tf.reduce_mean(s)  # sum over integrals of h_1, ..., h_d

    if penalty_method == 'exp':
        penal = 1/gamma * tf.reduce_mean(tf.exp(gamma * (f(x)-s) - 1))  # exponential penalty term
    elif penalty_method == 'L2':
        penal = gamma * tf.reduce_mean(tf.square(tf.nn.relu(f(x)-s)))  # L^2 penalty term
    else:
        raise ValueError('Your specified penalty method is not implemented')
    target_function = ints + penal

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(target_function)

    return x, gamma, train_op, target_function


def run_experiments(x, gamma, train_op, target_function, num_experiments, gamma_values, batch_values, n, final_n, d):
    """

    :param x: Input sample placeholder in tensorflow graph
    :param gamma: penalisation factor placeholder in tensorflow graph
    :param train_op: optimizer in tensorflow graph
    :param target_function: objective function in tensorflow graph
    :param num_experiments: number of numerical evaluations of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :param gamma_values: list of gamma values for different runs
    :param batch_values: list of batch size values for different runs
    :param n: number of iterations the network is trained
    :param final_n: final number of iterations that is averaged to obtain the value of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :param d: Dimension or number of marginals in the optimal transport problem. Each marginal has dimension 1
    :return: outputs and saves the experimental results
    """
    final_values = [0 for i in range(num_experiments)]
    start_time = time.time()

    # For the network structure and the hardware setup we used, running on a GPU was slower. Remove this config when
    # starting the sessions to run on GPU instead. Especially if you choose to increase the hidden dimension in the
    # network structure, GPU will scale better than CPU.
    config = tf.ConfigProto(
        device_count={'GPU': 0})

    for i in range(num_experiments):
        print(i)
        gamma_value = gamma_values[i]
        batch_size = batch_values[i]
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            gen_ = gen_points(batch_size)
            for t in range(n):
                input_sample = next(gen_)
                _ = sess.run([train_op], feed_dict={x: input_sample, gamma: [gamma_value]})
            c_fin = 0
            for o in range(final_n):
                input_sample = next(gen_)
                (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value]})
                c_fin += c/final_n
            final_values[i] = c_fin
            print(c_fin)

    runtime = time.time() - start_time
    print('Total runtime = ' + str(runtime))
    print('List of final values: ' + str(final_values))
    print('Mean of final values: ' + str(np.mean(final_values)))
    print('Variance of final values: ' + str(np.var(final_values)))
    print('Standard deviation of final values: ' + str(np.std(final_values)))

    final_values.append(runtime)
    directory = os.path.dirname(__file__)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt(directory + '/' + time_string + 'Dimension' + str(d) + '.txt', final_values)


if __name__ == '__main__':
    dim = 2  # Dimension
    num_experiments_1 = 100  # Number of times the sample value is calculated
    gamma_val = [80 for i in range(num_experiments_1)]  # gamma is always 80 for the table in the paper
    batch_val = [128 for i in range(num_experiments_1)]  # batch size is always 128 for the table in the paper
    n_1 = 29000  # Number of iterations
    final_n_1 = 1000  # Number of final iterations used to calculate the sample value of $\hat{\phi}_{\mu,\gamma}^m(f)$

    # Specify the parameters of the function f. I.e. z_value is the parameter z in the paper.
    # In difference to the paper, the distribution here are on [-0.5,0.5]^d, and not on [0,1]^d.
    # Hence the z values have to be adjusted accordingly
    z_val = [1 - 0.5] * dim
    z_val[0] = 0.5 - 0.5
    z_val[1] = 0.75 - 0.5

    # As penalty_method, currently only L^2 or exponential are included. New penalization methods have to be coded in
    # the build_graph function (the variable 'penal').
    penalty_method_1 = 'L2'
    # penalty_method_1 = 'exp'

    X, gamma_1, train_op_1, target_function_1 = build_graph(dimension=dim, penalty_method=penalty_method_1,
                                                            z_value=z_val)

    run_experiments(x=X, gamma=gamma_1, train_op=train_op_1, target_function=target_function_1,
                    num_experiments=num_experiments_1, gamma_values=gamma_val, batch_values=batch_val, n=n_1,
                    final_n=final_n_1, d=dim)
