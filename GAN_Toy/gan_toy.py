#!/usr/bin/env python

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import GAN_Toy.tflib as lib
import GAN_Toy.tflib.ops.linear
import GAN_Toy.tflib.plot

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets

# CHANGES #
# The main changes of this program compared to https://github.com/igul222/improved_wgan_training are:
# - Adjusted code to Python 3
# - Changed Parameters of the Adam optimizer to default (as used for all other programs)
# - Reduced hidden dimension of the network used by 1/2, as we use two function in the dual representation instead of 1
#   this keeps the number of parameters comparable to the original setting
# - implementing a different penalization method (one sided)

# - The standard Adam parameter used in all other programs don't work so well in an adversarial setting
#   as they are too high. So we kept the parameters from the initial program. For a fixed generator, one can adjust
#   them for higher accuracy of the transport problem
# - Further, in an adversarial setting, a higher number of CRITIC_ITERS is needed with one sided penalization,
#   in particular for the more exotic cost functions c.

# - Instead of plotting the contour of the Lipschitz function, the sum of the functions from the dual are plotted

# - allowing for different cost functions c in the optimal transport GAN as described in the paper. Some choices:
# c = 1  # the standard Wasserstein 1 GAN with euclidean norm to compare the toy problems to the original paper.
# c = 2  # Second order Wasserstein GAN with euclidean norm
# c = 3  # Second order Wasserstein GAN with L^1 norm.
# c = 4  # c(x,y) = exp(|x_1 - y_1|) + exp(|x_2 - y_2|) - Doesn't work well
# c = 5  # c(x,y) = <x, y>  - Doesn't work well
c = 6  # c(x,y) = min(1, max(0, |x-y| - 0.1) * 10)  - Interesting for fixed generator, useless (because bounded) in an
#                                                     adversarial setting. One would have to bound the network output.
# c = 7  # c(x,y) = sigmoid(10 * |x-y|) - Similar to c = 6.
gamma_value = 20  # The penalization factor. Here, we always use L^2 penalization.
# END CHANGES #

MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = '8gaussians' # 8gaussians, 25gaussians, swissroll
DIM = 512 # Model dimensionality
DIM2 = 256  # Model dimensionality for two functions in dual representation
FIXED_GENERATOR = True  # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 20 # How many critic iterations per generator iteration
BATCH_SIZE = 256 # Batch size
ITERS = 100000 # how many generator iterations to train for

lib.print_model_settings(locals().copy())

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples, real_data):
    if FIXED_GENERATOR:
        return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    else:
        noise = tf.random_normal([n_samples, 2])
        output = ReLULayer('Generator.1', 2, DIM, noise)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.4', DIM, 2, output)
        return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, DIM2, inputs)
    output = ReLULayer('Discriminator.2', DIM2, DIM2, output)
    output = ReLULayer('Discriminator.3', DIM2, DIM2, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM2, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE, real_data)

# The following is changed from the initial program:
def Discriminator_2(inputs):
    output = ReLULayer('Discriminator.5', 2, DIM2, inputs)
    output = ReLULayer('Discriminator.6', DIM2, DIM2, output)
    output = ReLULayer('Discriminator.7', DIM2, DIM2, output)
    output = lib.ops.linear.Linear('Discriminator.8', DIM2, 1, output)
    return tf.reshape(output, [-1])

disc_real2 = Discriminator(real_data)
fplot = Discriminator(real_data) + Discriminator_2(real_data)
disc_fake2 = Discriminator_2(fake_data)


# Note that the terms are always negative as the discriminator wants to maximize
if c == 1:
    c_term = -tf.sqrt(tf.reduce_sum(tf.square(real_data-fake_data), 1))
elif c == 2:
    c_term = -tf.reduce_sum(tf.square(real_data-fake_data), 1)
elif c == 3:
    c_term = -tf.square(tf.reduce_sum(tf.abs(real_data-fake_data), 1))
elif c == 4:
    c_term = -tf.exp(tf.abs(real_data[:, 0:1] - fake_data[:, 0:1])) - tf.exp(tf.abs(real_data[:, 1:2] -
                                                                                    fake_data[:, 1:2]))
elif c == 5:
    c_term = -tf.reduce_sum(tf.multiply(real_data, fake_data), 1)
elif c == 6:
    c_term = -(-tf.nn.relu(-tf.nn.relu(10 * (tf.sqrt(tf.reduce_sum(tf.square(real_data-fake_data), 1)) - 0.1)) + 1)+1)
elif c == 7:
    c_term = -tf.nn.sigmoid(10 * tf.sqrt(tf.reduce_sum(tf.square(real_data-fake_data), 1)))
else:
    c_term = 0

penal = gamma_value * tf.reduce_mean(tf.square(tf.nn.relu(c_term - Discriminator(real_data) -
                                                          Discriminator_2(fake_data))))
disc_cost2 = tf.reduce_mean(disc_fake2) + tf.reduce_mean(disc_real2) + penal
gen_cost2 = - tf.reduce_mean(disc_fake2) - penal

# End of code that was changed. In the following we just replaced disc_cost by disc_cost2, and gen_cost by gen_cost2

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient penalty
if MODE == 'wgan-gp':
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    interpolates = alpha*real_data + ((1-alpha)*fake_data)
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
    disc_cost += LAMBDA*gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE == 'wgan-gp':
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
        # beta1=0.99,
        # beta2=0.995
    ).minimize(
        disc_cost2,
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            gen_cost2,
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost2,
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost2,
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

print("Generator params:")
for var in lib.params_with_name('Generator'):
    print("\t{}\t{}".format(var.name, var.get_shape()))
print("Discriminator params:")
for var in lib.params_with_name('Discriminator'):
    print("\t{}\t{}".format(var.name, var.get_shape()))

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run(
        [fake_data, disc_real], 
        feed_dict={real_data:points}
    )
    disc_map = session.run(fplot, feed_dict={real_data:points})

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    if not FIXED_GENERATOR:
        plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')

    plt.savefig('frame'+str(frame_index[0])+'.jpg')
    frame_index[0] += 1

# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':
    
        dataset = []
        for i in range(round(100000/25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in range(round(len(dataset)/BATCH_SIZE)-1):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE, 
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield data

    elif DATASET == '8gaussians':
    
        scale = 2.
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield dataset

# Train loop!
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()
    for iteration in range(ITERS):
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={real_data: _data})
        # Train critic
        for i in range(CRITIC_ITERS):
            _data = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost2, disc_train_op],
                feed_dict={real_data: _data}
            )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])
        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        if iteration % 100 == 99:
            lib.plot.flush()
            generate_image(_data)
        lib.plot.tick()
