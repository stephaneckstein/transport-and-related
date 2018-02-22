import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from operator import add
sns.set(color_codes=True)
import os
import time


# Objective function
def g(p, th, lower_greater):
    sgf = tf.reduce_sum(p, 1)
    cond = tf.greater(lower_greater*sgf, lower_greater*th)
    return tf.cast(cond, dtype=tf.float64)


# rebuilds the above objective function in numpy
def g_rebuild(p, th, lower_greater):
    sgf = np.sum(p, axis=1)
    cond = np.greater(lower_greater*sgf, lower_greater*th)
    return cond.astype(np.float64)


def density_rebuild(w, x, gam_val, d, th, lower_greater):  # density as for L2 penalization, i.e. 2 * gam_val * x_+
    """

    :param w: weights that were read out of the neural network that define the optimal numerical solution
    :param x: input to the function, numpy list of size [~, d] where ~ is arbitrary.
    :param gam_val: gamma value for the density
    :param d: dimension
    :return: evaluates the density $\frac{d\nu^*}{d\mu}$ of an optimizer $\nu^*$ at point x.
    """
    s_rebuild = 0
    for k in range(d):
        s_rebuild += univ_rebuild(w[7 * k:7 * (k + 1)], [[x[j, k]] for j in range(len(x))])
    return 2 * gam_val * np.maximum(g_rebuild(x, th, lower_greater) - s_rebuild, 0)


# Programmed for multiple updating steps. We found this to be clearly limited by the numerical implementation, and hence
# only included a single updating step in the rest of the program.
def up_gen_eff(vector_w, gam_val, th, batch_size, d, lower_greater, batch_up=2 ** 21, par=0):
    full_sample = []
    while 1:
        if par < 2:
            raw_sample = np.random.random_sample([batch_up, d])
        else:
            raw_sample = np.random.pareto(par, [batch_up, d])
        sample_up_n = raw_sample
        for i in range(len(vector_w)):
            sample_up = sample_up_n[:]
            dens_eval = density_rebuild(vector_w[i], sample_up, gam_val, d, th, lower_greater)

            # max_sample = np.percentile(dens_eval, 99.9)  # Changes distribution but reduces number of rejections
            max_sample = np.amax(dens_eval)

            unif_sample = np.random.random_sample(size=[batch_up]) * max_sample
            sample_up_n = []
            for j in range(len(sample_up)):
                if dens_eval[j] >= unif_sample[j]:
                    sample_up_n.append(sample_up[j])
        full_sample.extend(sample_up_n)
        while len(full_sample) > batch_size:
            out = np.zeros([batch_size, d])
            for j in range(batch_size):
                x = full_sample.pop()
                for k in range(d):
                    out[j, k] = x[k]
            yield (out)


def univ_rebuild(w, x):
    z = np.matmul(x, w[0])
    z = z + w[1]
    a = np.maximum(z, 0)
    z2 = np.matmul(a, w[2]) + w[3]
    a2 = np.maximum(z2, 0)
    z3 = np.matmul(a2, w[4]) + w[5]
    a3 = np.maximum(z3, 0)
    return np.sum(np.matmul(a3, w[6]), axis=1)


def gen_points(batch_size, d):
    while True:
        dataset = np.random.random_sample([batch_size, d])
        yield(dataset)


def gen_points_pareto(batch_size, d, par=2.3):
    while True:
        dataset = np.random.pareto(par, [batch_size, d])
        yield(dataset)


# General layer structure to approximate the univariate functions h_1, ..., h_d
def univ_approx(x, name, hidden_dim=64):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable('ua_w1', shape=[1, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        ua_b = tf.get_variable('ua_b1', shape=[hidden_dim],
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


def build_graph(th, lower_greater, d=2):
    x = tf.placeholder(dtype=tf.float64, shape=[None, d])  # Random samples
    gamma = tf.placeholder(dtype=tf.float64, shape=[1])  # Risk Aversion factor
    x_marginal = tf.placeholder(dtype=tf.float64, shape=[None, d])

    s = 0
    for i in range(d):
        s += univ_approx(x[:, i:i + 1], str(i))

    s2 = 0
    for i in range(d):
        s2 += univ_approx(x_marginal[:, i:i + 1], str(i))
    ints = tf.reduce_mean(s2)
    target_function = ints + gamma * tf.reduce_mean(tf.square(tf.nn.relu(g(x, th, lower_greater) - s)))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(target_function)

    # The below is a very quick and dirty implementation to only train the last layer with optimizer 2.
    # In the last few iterations, we will only train the last layer of the functions h_1 and h_2 and average the weights
    # of these functions when reading out the optimal function \hat{h}. This is done in an attempt to make the
    # numerical process more stable.
    listofnames = [str(i) + '/ua_v:0' for i in range(d)]
    variable_list = [v for v in tf.global_variables() if v.name in listofnames]
    print(variable_list)
    train_op_fix = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(target_function,
                                                                                                var_list=variable_list)
    return x, x_marginal, gamma, target_function, train_op, train_op_fix


def run_experiments(x, x_marginal, gamma, train_op, train_op_fix, target_function, gamma_value=320, batch_size=2**10,
                    n=20000, n_update=10000, final_n=1000, final_update=500, d=2, th=1.9, lower_greater=1, par=0):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    saved_weights = []
    with tf.Session(config=config) as sess:
        # First run without update
        sess.run(tf.global_variables_initializer())

        if par < 2:
            gen_ = gen_points(batch_size, d)
        else:
            gen_ = gen_points_pareto(batch_size, d, par=par)

        for t in range(n):
            input_sample = next(gen_)
            (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value],
                                                                      x_marginal: input_sample})
            if t % 2000 == 0:
                print(c)
        final_value = 0
        w1_val = sess.run([v.name for v in tf.trainable_variables()])
        w1_val[:] = [w1x / (final_n + 1) for w1x in w1_val]
        for t in range(final_n):
            input_sample = next(gen_)
            (_, c) = sess.run([train_op_fix, target_function], feed_dict={x: input_sample, gamma: [gamma_value],
                                                                       x_marginal: input_sample})
            w2_val = sess.run([v.name for v in tf.trainable_variables()])
            w2_val[:] = [w2x / (final_n + 1) for w2x in w2_val]
            w1_val[:] = map(add, w1_val, w2_val)

            final_value += c / final_n
        print('Final value without Update = ' + str(final_value))
        saved_weights.append(w1_val)

        # Running for the updated reference measure after plotting a sample from the marginals
        sess.run(tf.global_variables_initializer())
        gen_ = up_gen_eff(saved_weights, gamma_value, th, batch_size, d, lower_greater, par=par)
        if par < 2:
            gen_marginal = gen_points(batch_size, d)
        else:
            gen_marginal = gen_points_pareto(batch_size, d, par=par)

        plot_sample = next(gen_)
        df = pd.DataFrame(plot_sample, columns=["x", "y"])
        for pi in range(25):
            plot_sample_app = next(gen_)
            dfa = pd.DataFrame(plot_sample_app, columns=["x", "y"])
            df = df.append(dfa)
        sns.jointplot(x="x", y="y", data=df, xlim=[0, 1], ylim=[0, 1], s=1, stat_func=None)
        # sns.jointplot(x="x", y="y", data=df, s=1, stat_func=None)

        directory = os.path.dirname(__file__)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(directory + '/' + time_string + '.jpg', format='jpg', dpi=600)
        plt.show()

        for t in range(n_update):
            input_sample = next(gen_)
            input_marginal = next(gen_marginal)
            (_, c) = sess.run([train_op, target_function], feed_dict={x: input_sample, gamma: [gamma_value],
                                                                       x_marginal: input_marginal})
            if t % 1000 == 0:
                print(c)
        final_value = 0
        for t in range(final_update):
            input_sample = next(gen_)
            input_marginal = next(gen_marginal)
            # input_marginal = input_sample
            (_, c) = sess.run([train_op_fix, target_function], feed_dict={x: input_sample, gamma: [gamma_value],
                                                                       x_marginal: input_marginal})
            final_value += c / final_update
        print('Final value after one update = ' + str(final_value))

if __name__ == '__main__':

    dim = 2  # Dimension.

    # Parameters to define the objective function f, for which holds:
    # f(x) = 1, if (lower_greater * sum_{i=1}^d x_i) >= (lower_greater * th), and f(x) = 0, else.
    # I.e. if lower_greater is one, it is >=, if lower_greater is -1, it is <=
    th_value = 1.9
    lower_greater_value = 1

    par_value = 0  # Means the marginals are uniform on [0,1]. If par_value > 2, then
    # it is taken as the parameter of a Pareto distribution and all marginals are sampled from the Pareto
    # (or Lomax) distribution with parameter par_value as implemented by numpy.

    gamma_value = 320  # Penalty factor
    batch_size = 2 ** 10
    n = 20000  # Number of iterations in the first optimization before updating the reference measure
    n_update = 10000  # Number of iterations in the optimization after updating. Less is sufficient here
    final_n = 1000  # Number iterations to calculate $\hat{\phi}_{\mu,\gamma}^m(f)$
    final_update = 500  # Number iterations to calculate $\hat{\phi}_{\nu^*,\gamma}^m(f)$ (updated reference measure)

    X, X_marginal, gamma_1, target_function_1, train_op_1, train_op_fix_1 = build_graph(th_value, lower_greater_value,
                                                                                        d=dim)

    run_experiments(x=X, x_marginal=X_marginal, target_function=target_function_1, train_op=train_op_1, gamma=gamma_1,
                    train_op_fix=train_op_fix_1, th=th_value, lower_greater=lower_greater_value, d=dim, par=0,
                    gamma_value=gamma_value, batch_size=batch_size, n=n, n_update=n_update, final_n=final_n,
                    final_update=final_update)
