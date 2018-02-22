import tensorflow as tf
import numpy as np
import time
import os

d = 2  # Dimension. Fix here, since the program would be a bit different with more assets.


# Sample from $\mu^{(1)}
def gen_points(batch_size):
    while True:
        dataset = np.zeros([batch_size, d])
        dataset[:, 0] = np.random.random_sample(batch_size)
        dataset[:, 1] = 2 * ((np.random.random_sample(batch_size)) ** 2)
        yield(dataset)


# Sample from $\mu^{(2)}$
def gen_points_1(batch_size):
    while True:
        dataset = np.zeros([batch_size, d])
        dataset[:, 0] = np.random.random_sample(batch_size)
        dataset[:round(batch_size/2), 1] = 2 * ((np.random.random_sample(round(batch_size/2))) ** 2)
        dataset[round(batch_size/2):, 1] = 2 * (dataset[round(batch_size/2):, 0] ** 2)
        yield (dataset)


# General layer structure to approximate the uni-variate functions f_1,...,f_d
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
    return z


def build_graph(penalty_method='L2'):
    def penalization_variance(p):
        p_mean = tf.reduce_mean(p)
        return (p-p_mean)**2

    def f(p):
        return p - penalization_factor * penalization_variance(p)

    x = tf.placeholder(dtype=tf.float32, shape=[None, d])  # Random samples
    gamma = tf.placeholder(dtype=tf.float32, shape=[1])  # penalty factor
    penalization_factor = tf.placeholder(dtype=tf.float32, shape=[1])  # risk aversion factor
    w = tf.get_variable('w', initializer=tf.constant(0.5))  # Fraction of Portfolio invested in first asset
    xw = (1-w) * x[:, 0:1] + w * x[:, 1:2]
    w_penal = 100 * tf.nn.relu(-w) + 100 * tf.nn.relu((w-1))  # Enforce that the weights are between 0 and 1.

    s = 0
    for i in range(d):
        s += univ_approx(x[:, i:i+1], str(i))
    ints = tf.reduce_mean(s)  # sum over integrals of f_i

    target_function = ints + gamma * tf.reduce_mean(tf.square(tf.nn.relu(-f(xw)-s))) + w_penal
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(target_function)

    return x, gamma, train_op, target_function, w, penalization_factor


def run_experiments(x, gamma, train_op, target_function, w, penalization_factor, num_experiments=11, n=40000,
                    pen_values=[0.1*i for i in range(11)], gamma_value=160, batch_size=2**13, final_n=1000,
                    reference_measure='Product'):
    """

    :param x: Input sample placeholder in tensorflow graph
    :param gamma: penalisation factor placeholder in tensorflow graph
    :param train_op: optimizer in tensorflow graph
    :param target_function: objective function in tensorflow graph
    :param w: portfolio weight of second asset in tensorflow graph
    :param penalization_factor: penalization or risk aversion factor in tensorflow graph
    :param num_experiments: number of different penalization factors evaluated
    :param n: number of iterations of training optimizer
    :param pen_values: list of penalization or risk aversion values for each experiment
    :param gamma_value: penalty factor for $\beta_{\gamma}$
    :param batch_size: single batch size for all experiments here
    :param final_n: final number of iterations that is averaged to obtain the value of $\hat{\phi}_{\mu,\gamma}^m(f)$
    :param reference_measure: Either 'Product' or 'Correlated'. Specifies the reference measure $\mu$.
    :return:
    """

    config = tf.ConfigProto(
            device_count={'GPU': 0}
    )

    value_list = []
    weight_list = []
    for ii in range(num_experiments):
        penalization_value = pen_values[ii]
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if reference_measure == 'Product':
                gen_ = gen_points(batch_size)  # For sampling from the product reference measure
            elif reference_measure == 'Correlated':
                gen_ = gen_points_1(batch_size)  # For sampling from the correlated reference measure
            else:
                raise ValueError('Your chosen reference measure is not implemented')
            cav = 1
            wav = 0.5
            for t in range(n):
                input_sample = next(gen_)
                (_, c, wv) = sess.run([train_op, target_function, w], feed_dict={x: input_sample, gamma: [gamma_value],
                                                                            penalization_factor: [penalization_value]})
                if t % 1000 == 0:
                    print(c)

            c_fin = 0
            w_fin = 0
            for o in range(final_n):
                input_sample = next(gen_)
                (_, c, wv) = sess.run([train_op, target_function, w], feed_dict={x: input_sample,
                                                      gamma: [gamma_value], penalization_factor: [penalization_value]})
                c_fin += c/final_n
                w_fin += wv/final_n
            print("Factor " + str(penalization_value) + " has value " + str(c_fin) + " and weight " + str(w_fin))
            value_list.append(c_fin)
            weight_list.append(w_fin)

    time_string = time.strftime("%Y%m%d-%H%M%S")
    directory = os.path.dirname(__file__)
    np.savetxt(directory + '/Values' + reference_measure + time_string + '.txt', value_list)
    np.savetxt(directory + '/Weights' + reference_measure + time_string + '.txt', weight_list)


if __name__ == '__main__':
    # Specify which measure to sample from:
    # reference_measure_1 = 'Product'
    reference_measure_1 = 'Correlated'

    num_experiments = 11  # Number of different risk aversion factors that the program is executed for
    n = 40000  # Number of iterations for each experiment
    pen_values = [0.1 * i for i in range(num_experiments)]  # Risk aversion factors of the investor
    gamma_value = 160  # Penalty factor
    batch_size = 2 ** 13
    final_n = 1000  # Number of final iterations used to calculate the sample value of $\hat{\phi}_{\mu,\gamma}^m(f)$

    X, gamma_1, train_op_1, target_function_1, w_1, penalization_factor_1 = build_graph()
    run_experiments(x=X, gamma=gamma_1, target_function=target_function_1, train_op=train_op_1, w=w_1,
                    penalization_factor=penalization_factor_1, reference_measure=reference_measure_1, num_experiments=
                    num_experiments, n=n, pen_values=pen_values, gamma_value=gamma_value, batch_size=batch_size,
                    final_n=final_n)
