#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def running_mean(l, k):
    """ Function to plot the running k-mean of a list l of values

    :param l: list of numbers
    :param k: integer
    :return
    """
    cumulative_sum = np.cumsum(np.insert(l, 0, 0))
    return (cumulative_sum[k:] - cumulative_sum[:-k]) / float(k)


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
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z)
        ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        z2 = tf.matmul(a, ua_w2) + ua_b2
        a2 = tf.nn.relu(z2)
        ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        z3 = tf.matmul(a2, ua_w3) + ua_b3
        a3 = tf.nn.relu(z3)
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


def build_graph(dimension, penalty_method='L2', z_value=(0, 0.25)):
    """Specify network structure

    :param dimension: Here: Number of marginals (i.e. dimension of the whole space). Each marginal has dimension 1
    :param penalty_method: Specifies which penalty to use: Currently 'L2' or 'exp'
    :param z_value: Parameter for the function f
    :return: Various elements of the graph.
    """

    def f(p):
        """ f(p) = 1 if p <= z_value for all elements of p, and f(p) = 0, else

        :param p: tf variable with size [batch_size, dimension] of type float
        :return: tf variable with size [batch_size, 1] of type tf.float64
        """
        cond1 = tf.greater(p, z_value)
        cond1 = tf.reduce_any(cond1, 1)
        cond1 = tf.logical_not(cond1)
        out = tf.cast(cond1, dtype=tf.float64)
        return out

    x = tf.placeholder(dtype=tf.float64, shape=[None, dimension])  # Samples from $\mu$, also used for marginals
    gamma = tf.placeholder(dtype=tf.float64, shape=[1])  # penalty factor

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


def run_experiments(z_value, x, gamma, train_op, target_function, num_experiments=4, gamma_values=(80, 80, 10, 10),
                    batch_values=(128, 2048, 128, 2048), mean_n=1000, n=50000, penalty_method='L2', final_n=1000):
    """

    :param z_value: specifies function f.
    :param x: Input sample placeholder in tensorflow graph
    :param gamma: penalisation factor placeholder in tensorflow graph
    :param train_op: optimizer in tensorflow graph
    :param target_function: objective function in tensorflow graph
    :param num_experiments: number of numerical evaluations of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :param gamma_values: list of gamma values for different runs
    :param batch_values: list of batch size values for different runs
    :param mean_n: plots are running averages over mean_n many iterations of the training
    :param n: number of iterations in the training
    :param penalty_method: specifies the function $\beta_{\gamma}$ for the penalty method
    :param final_n: final number of iterations that is averaged to obtain the value of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :return: outputs and saves the experimental results
    """
    run_times = [0 for i in range(num_experiments)]
    final_values = [0 for i in range(num_experiments)]
    plt.figure(figsize=(8, 6))

    # For the network structure and the hardware setup we used, running on a GPU was slower. Remove this config when
    # starting the sessions to run on GPU instead. Especially if you choose to increase the hidden dimension in the
    # network structure, GPU will scale better than CPU.
    config = tf.ConfigProto(
        device_count={'GPU': 0})

    for i in range(num_experiments):
        gamma_value = gamma_values[i]
        batch_size = batch_values[i]
        value_list = []
        with tf.Session(config=config) as sess:
            start_time = time.time()
            init = tf.global_variables_initializer()
            sess.run(init)
            gen_ = gen_points(batch_size=batch_size)
            for t in range(1, n+1):
                input_sample = next(gen_)
                (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value]})
                value_list.append(min(max(c, -5), 5))  # To avoid rare crazy explosions with exponential penalty
                if t % 1000 == 0:
                    print(c)
                    print(t)

            runtime = time.time() - start_time
            run_times[i] = runtime
            print(runtime)

            # As value for $\hat{\phi}_{\mu,\gamma}^m(f)$ we finally take the average over final_n many iterations
            c_fin = 0
            for o in range(final_n):
                input_sample = next(gen_)
                (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value]})
                c_fin += c/final_n
            final_values[i] = c_fin

            # We plot the running_means of the values at each iteration.
            # Possible TODO: Extract plot method
            plot_list = running_mean(value_list, mean_n)
            ta = np.arange(1, t+2-mean_n, 1)
            real_value = min(z_value)+0.5
            plt.plot(ta, np.ones(t+1-mean_n) * real_value, 'r--')

            # In the If-statement the (probably) best-possible bounds from below that can be obtained analytically
            # are plotted.
            if penalty_method == 'exp':
                plt.plot(ta, np.ones(t+1-mean_n) * (real_value - 1/gamma_value * 0.2158), 'b--')
            elif penalty_method == 'L2':
                plt.plot(ta, np.ones(t+1-mean_n) * (real_value - 1/gamma_value * 1/3), 'b--')
            else:
                print('Your specified penalty method does not have analytical bounds from below hardcoded')
            plt.plot(ta, plot_list, label='$\gamma =$'+str(gamma_value)+'; ' + 'batch: ' + str(batch_size))
            plt.legend()
            plt.ylabel('$\hat{\phi}_{\mu,\gamma}^{m}(f)$')
            plt.xlabel('iterations')
            axes = plt.gca()
            axes.set_ylim([real_value-0.1, real_value+0.1])

    print('Final values for each individual experiment:' + str(final_values))
    print('Run times for each individual experiment:' + str(run_times))
    if penalty_method == 'exp':
        plt.title('Exponential Penalization')
    elif penalty_method == 'L2':
        plt.title('$L^2$ Penalization')
    else:
        plt.title(penalty_method + 'Penalization')
    directory = os.path.dirname(__file__)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    if penalty_method == 'exp':
        filename = directory + '/ExponentialHQ' + time_string + '.jpg'
    elif penalty_method == 'L2':
        filename = directory + '/L2HQ' + time_string + '.jpg'
    else:
        filename = directory + '/' + penalty_method + time_string + '.jpg'
    plt.savefig(filename, format='jpg', dpi=700)
    plt.show()


if __name__ == '__main__':
    d = 2  # Dimension
    num_experiments_1 = 4  # Specifies how many different values for $\gamma$ and batch_size are calculated/plotted
    gamma_val = (80, 80, 10, 10)  # These are the gamma values for the L^2 penalization in the paper
    # gamma_val = (40, 40, 10, 10)  # These are the gamma values for the exponential penalization in the paper
    batch_val = (128, 2048, 128, 2048)  # batch sizes used for the individual experiments
    meanN = 1000  # This specifies over how many iterations the running mean is taken in the plot.
    n_1 = 50000  # Number of iterations
    final_n_1 = 1000  # Number of final iterations used to calculate the sample value of $\hat{\phi}_{\mu,\gamma}^m(f)$

    # Specify the parameters of the function f. I.e. z_value is the parameter z in the paper.
    z_val = [0.5, 0.75]
    z_val = [z_val[0] - 0.5, z_val[1] - 0.5]  # We centralize all measure from [0,1] to [-0.5,0.5] -> shift z

    # As penalty_method, currently only L^2 or exponential are included. New penalization methods have to be coded in
    # the build_graph section (the variable 'penal').
    penalty_method_1 = 'L2'
    # penalty_method_1 = 'exp'

    X, gamma_1, train_op_1, target_function_1 = build_graph(dimension=d, penalty_method=penalty_method_1, z_value=z_val)

    run_experiments(z_value=z_val, x=X, gamma=gamma_1, train_op=train_op_1, target_function=target_function_1,
                    num_experiments=num_experiments_1, gamma_values=gamma_val, batch_values=batch_val,
                    mean_n=meanN, n=n_1, penalty_method=penalty_method_1, final_n=final_n_1)


