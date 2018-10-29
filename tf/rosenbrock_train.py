"""Script to train Rosenbrock function MLP approximation.

A simple usage example:
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=lm -kwargs={'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':10} -out=log1.txt
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=sgd -kwargs={'learning_rate':1e-3} -out=log2.txt
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=adam -kwargs={'learning_rate':1e-3} -out=log3.txt
"""
import argparse
import numpy as np
import tensorflow as tf
import time
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    '-N', help='Rosenbrock function dimensionality',
    type=int, default=4)
parser.add_argument(
    '-m', help='number of random data points',
    type=int, default=5000)
parser.add_argument(
    '-hidden', help='MLP hidden layers structure',
    type=str, default='[8,4]')
parser.add_argument(
    '-a', '--activation', help='nonlinear activation function',
    type=str, choices=['relu', 'sigmoid', 'tanh'], default='tanh')
parser.add_argument(
    '-i', '--initializer', help='trainable parameters initializer',
    type=str, choices=['rand_normal', 'rand_uniform', 'xavier'],
    default='xavier')
parser.add_argument(
    '-opt', '--optimizer', help='optimization algorithms',
    type=str, choices=['sgd', 'adam', 'lm'], default='lm')
parser.add_argument(
    '-kwargs', help='optimizer parameters',
    type=str, default="{'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':10}")
parser.add_argument(
    '-out', help='output stat file name',
    type=str, default='log.txt')

SEED = 10

ACTIVATIONS = {'tanh': tf.nn.tanh,
               'relu': tf.nn.relu,
               'sigmoid': tf.nn.sigmoid}

INITIALIZERS = {'xavier': tf.contrib.layers.xavier_initializer(seed=SEED),
                'rand_uniform': tf.random_uniform_initializer(seed=SEED),
                'rand_normal': tf.random_normal_initializer(seed=SEED)}

TF_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer,
                 'adam': tf.train.AdamOptimizer}

DT = tf.float64
DT_NP = np.float64

TARGET_LOSS = 1e-10
MAX_STEPS = 400000000

LOG_INTERVAL_IN_SEC = 10

log_file_name = None
log_prev_time, log_first_time = None, None


def log(step, loss):
    global log_prev_time, log_first_time

    now = time.time()
    if log_prev_time and now-log_prev_time < LOG_INTERVAL_IN_SEC:
        return
    if not log_prev_time:
        log_prev_time, log_first_time = now, now
    message = f'{step} {int(now-log_first_time)} {loss}'
    print(message)
    with open(log_file_name, "a") as file:
        file.write(message+'\n')
    log_prev_time = now


def get_rand_rosenbrock_data_points(n, m, x_range, x_norm=None, y_norm=None):
    np.random.seed(SEED)
    x = np.random.uniform(*x_range, (m, n))
    y = np.zeros(shape=(m, 1), dtype=DT_NP)
    # include local and global minima for 4 <= N <= 7
    x[0, :] = ([-1] + [1] * (n - 1))
    x[1, :] = [1] * n
    for s in range(m):
        for i in range(n - 1):
            y[s] += 100*((x[s, i+1]-x[s, i]**2)**2) + (1-x[s,i])**2
    # do normalization
    if not x_norm:
        x_norm = max(x_range)
    if not y_norm:
        y_norm = max(y)[0]
    x = x/x_norm
    y = y/y_norm - 0.5
    return x, y, x_norm, y_norm


def jacobian(y, x, m):
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(DT, size=m),
    ]
    _, jacobian = tf.while_loop(
        lambda i, _: i < m,
        lambda i, res: (i+1, res.write(i, tf.squeeze(tf.gradients(y[i], x)))),
        loop_vars)
    return jacobian.stack()


def train_lm(feed_dict, params, y_hat, r, loss, mu, mu_inc, mu_dec, max_inc):
    neurons_cnt = params.shape[0].value
    m = y_hat.shape[0].value

    mu_current = tf.placeholder(DT, shape=[1])
    I = tf.eye(neurons_cnt, dtype=DT)
    y_hat_flat = tf.squeeze(y_hat)
    j = jacobian(y_hat_flat, params, m)
    j_t = tf.transpose(j)
    hess = tf.matmul(j_t, j)
    g = tf.matmul(j_t, r)
    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=DT))
    hess_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=DT))
    g_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=DT))
    save_params = tf.assign(p_store, params)
    restore_params = tf.assign(params, p_store)
    save_hess_g = [tf.assign(hess_store, hess), tf.assign(g_store, g)]
    dx = tf.matmul(tf.linalg.inv(hess_store + tf.multiply(mu_current, I)),
                   g_store)
    dx = tf.squeeze(dx)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1)
    lm = opt.apply_gradients([(-dx, params)])

    feed_dict[mu_current] = np.array([mu])
    session = tf.Session()
    step = 0
    session.run(tf.global_variables_initializer())
    current_loss = session.run(loss, feed_dict)
    while current_loss > TARGET_LOSS and step < MAX_STEPS:
        step += 1
        log(step, current_loss)
        session.run(save_params)
        session.run(save_hess_g, feed_dict)
        success = False
        for i in range(max_inc):
            session.run(lm, feed_dict)
            new_loss = session.run(loss, feed_dict)
            if new_loss < current_loss:
                feed_dict[mu_current] /= mu_dec
                current_loss = new_loss
                success = True
                break
            feed_dict[mu_current] *= mu_inc
            session.run(restore_params)
        if not success:
            print('Failed to improve')
            break


def tf_train(feed_dict, loss, train_step):
    step = 0
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calc initial loss
    current_loss = session.run(loss, feed_dict)
    while current_loss > TARGET_LOSS and step < MAX_STEPS:
        step += 1
        log(step, current_loss)
        session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)


def main():
    global log_file_name
    args = parser.parse_args()

    N, m = args.N, args.m
    initializer = INITIALIZERS[args.initializer]
    activation = ACTIVATIONS[args.activation]
    mlp_hidden_structure = eval(args.hidden)
    mlp_hidden_layers_cnt = len(mlp_hidden_structure)
    kwargs = eval(args.kwargs)
    log_file_name = args.out
    optimizer_name = args.optimizer

    with open(log_file_name, "a") as file:
        file.write(f'{" ".join(sys.argv[1:])}\n')

    dp_x, dp_y, x_norm, y_norm = get_rand_rosenbrock_data_points(N, m, (-2, 2))

    # placeholder variables (we have m data points)
    x = tf.placeholder(DT, shape=[m, N])
    y = tf.placeholder(DT, shape=[m, 1])
    feed_dict = {x: dp_x, y: dp_y}

    mlp_structure = [N] + mlp_hidden_structure + [1]
    tensors_shapes = []
    for i in range(len(mlp_hidden_structure)+1):
        tensors_shapes.append((mlp_structure[i], mlp_structure[i + 1]))
        tensors_shapes.append((1, mlp_structure[i + 1]))
    tensors_sizes = [h * w for h, w in tensors_shapes]
    neurons_cnt = sum(tensors_sizes)
    print(f'Total number of trainable parameters is {neurons_cnt}')

    params = tf.Variable(initializer([neurons_cnt], dtype=DT))
    tensors = tf.split(params, tensors_sizes, 0)
    for i in range(len(tensors)):
        tensors[i] = tf.reshape(tensors[i], tensors_shapes[i])
    Ws = tensors[0:][::2]
    bs = tensors[1:][::2]

    y_hat = x
    for i in range(mlp_hidden_layers_cnt):
        y_hat = activation(tf.matmul(y_hat, Ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, Ws[-1]) + bs[-1]

    r = y - y_hat
    loss = tf.reduce_mean(tf.square(r))

    if optimizer_name == 'lm':
        train_lm(feed_dict, params, y_hat, r, loss, **kwargs)
    else:
        train_step = TF_OPTIMIZERS[optimizer_name](**kwargs).minimize(loss)
        tf_train(feed_dict, loss, train_step)


main()
