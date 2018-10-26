"""Script to train MLP to approximate Rosenbrock function.

A simple usage example:
python rosenbrock_train.py -N=4 -m=5000 -hidden="[8,4]" -opt=lm -p="{'mu':3,'mu_inc':10,'mu_dec':10,'max_inc':10}" -out=log.txt
"""
import argparse
import numpy as np
import tensorflow as tf


ACTIVATIONS = {'tanh': tf.nn.tanh,
               'relu': tf.nn.relu,
               'sigmoid': tf.nn.sigmoid}


def generate_rosenbrock_data_points(N, m, x_range, x_norm=None, y_norm=None):
    x = np.random.uniform(*x_range, (m, N))
    y = np.zeros(shape=(m, 1), dtype=np.float64)
    # include local and global minima for 4 <= N <= 7 for checks
    x[0, :] = ([-1] + [1]*(N-1))
    x[1, :] = [1]*N
    for s in range(m):
        for i in range(N - 1):
            y[s] += 100*((x[s, i+1]-x[s, i]**2)**2) + (1-x[s,i])**2
    if not x_norm:
        x_norm = max(x_range)
    if not y_norm:
        y_norm = max(y)[0]
    x = x/x_norm
    y = y/y_norm - 0.5
    return x, y, x_norm, y_norm


def jacobian(y, x):
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float64, size=m),
    ]

    _, jacobian = tf.while_loop(
        lambda i, _: i < m,
        lambda i, res: (i+1, res.write(i, tf.squeeze(tf.gradients(y[i], x)))),
        loop_vars)

    return jacobian.stack()


parser = argparse.ArgumentParser()
parser.add_argument('-N', help='Rosenbrock function dimensionality', type=int, default=4)
parser.add_argument('-m', help='number of random data points', type=int, default=5000)
parser.add_argument('-hidden', help='MLP hidden layers structure', type=str, default='[8,4]')
parser.add_argument('-a', '--activation', help='nonlinear activation function', type=str,
                    choices=['relu', 'sigmoid', 'tanh'], default='tanh')
parser.add_argument('-i', '--initializer', help='trainable parameters initializer', type=str,
                    choices=['rand_normal', 'xavier'], default='xavier')
parser.add_argument('-opt', '--optimizer', help='optimization algorithms', type=str, choices=['sgd', 'adam', 'lm'],
                    default='lm')
parser.add_argument('-p', '--params', help='optimizer parameters', type=str,
                    default="{'mu':3,'mu_inc':10,'mu_dec':10,'max_inc':10}")
parser.add_argument('-out', help='output stat file name', type=str, default='log.txt')
args = parser.parse_args()


N, m = args.N, args.m
dp_x, dp_y, x_norm, y_norm = generate_rosenbrock_data_points(N, m, (-2, 2))

mlp_hidden_structure = eval(args.hidden)
mlp_full_structure = [N] + mlp_hidden_structure + [1]
tensors_shapes = []
for i in range(len(mlp_hidden_structure)+1):
    tensors_shapes.append((mlp_full_structure[i], mlp_full_structure[i + 1]))
    tensors_shapes.append((1, mlp_full_structure[i + 1]))
tensors_sizes = [h * w for h, w in tensors_shapes]
total_tensors_size = sum(tensors_sizes)

activation = ACTIVATIONS[args.activation]

# placeholder variables (we have m data points)
x = tf.placeholder(tf.float64, shape=[m, N])
y = tf.placeholder(tf.float64, shape=[m, 1])

p = tf.Variable(initializer([neurons_cnt], dtype=tf.float64))
parms = tf.split(p, sizes, 0)
for i in range(len(parms)):
    parms[i] = tf.reshape(parms[i], shapes[i])
Ws = parms[0:][::2]
bs = parms[1:][::2]

y_hat = x
for i in range(len(nn)):
    y_hat = activation(tf.matmul(y_hat, Ws[i]) + bs[i])
y_hat = tf.matmul(y_hat, Ws[-1]) + bs[-1]
y_hat_flat = tf.squeeze(y_hat)

r = y - y_hat
loss = tf.reduce_mean(tf.square(r))

mu = tf.placeholder(tf.float64, shape=[1])

p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=tf.float64))

save_parms = tf.assign(p_store, p)
restore_parms = tf.assign(p, p_store)

I = tf.eye(neurons_cnt, dtype=tf.float64)

j = jacobian(y_hat_flat, p)
jT = tf.transpose(j)
jTj = tf.matmul(jT, j)
jTr = tf.matmul(jT, r)

jTj_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=tf.float64))
jTr_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=tf.float64))
save_jTj_jTr = [tf.assign(jTj_store, jTj), tf.assign(jTr_store, jTr)]

dx = tf.matmul(tf.linalg.inv(jTj_store + tf.multiply(mu, I)), jTr_store)
dx = tf.squeeze(dx)
lm = opt.apply_gradients([(-dx, p)])