"""Script to train Rosenbrock function MLP approximation.

A simple usage example:
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=lm \
                           -kwargs={'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':10} \
                           -out=log1.txt
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=sgd \
                           -kwargs={'learning_rate':1e-3} -out=log2.txt
python rosenbrock_train.py -N=7 -m=20000 -hidden=[16,12,8] -opt=adam \
                           -kwargs={'learning_rate':1e-3} -out=log3.txt
"""
import argparse
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import os.path


# with fixed seed initial values for trainable variables and training data
# will be the same, so it is easier to compare optimization performance
SEED = 10
# you can try tf.float32/np.float32 data types
TF_DATA_TYPE = tf.float64
NP_DATA_TYPE = np.float64
# how frequently log is written and checkpoint saved
LOG_INTERVAL_IN_SEC = 10

# variants of activation functions
ACTIVATIONS = {'tanh': tf.nn.tanh,
               'relu': tf.nn.relu,
               'sigmoid': tf.nn.sigmoid}

# variants of initializers
INITIALIZERS = {'xavier': tf.contrib.layers.xavier_initializer(seed=SEED),
                'rand_uniform': tf.random_uniform_initializer(seed=SEED),
                'rand_normal': tf.random_normal_initializer(seed=SEED)}

# variants of tensorflow built-in optimizers
TF_OPTIMIZERS = {'sgd': tf.train.GradientDescentOptimizer,
                 'adam': tf.train.AdamOptimizer}

# checkpoints are saved to <log_file_name>.ckpt
out_file = None
log_prev_time, log_first_time = None, None
# are used to continue log when script is started from a checkpoint
step_delta, time_delta = 0, 0


def parse_arguments():
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
        '-i', '--initializer', help='trainable variables initializer',
        type=str, choices=['rand_normal', 'rand_uniform', 'xavier'],
        default='xavier')
    parser.add_argument(
        '-opt', '--optimizer', help='optimization algorithms',
        type=str, choices=['sgd', 'adam', 'lm'], default='lm')
    parser.add_argument(
        '-kwargs', help='optimizer parameters',
        type=str, default="{'mu':3.,'mu_inc':10,'mu_dec':10,'max_inc':10}")
    parser.add_argument(
        '-out', help='output log file name',
        type=str, default='log.txt')
    parser.add_argument("-c", "--cont", help="continue from checkpoint",
                        action="store_true")
    args = parser.parse_args()
    n = args.N
    m = args.m
    hidden = eval(args.hidden)
    activation = ACTIVATIONS[args.activation]
    initializer = INITIALIZERS[args.initializer]
    optimizer = args.optimizer
    kwargs = eval(args.kwargs)
    out = args.out
    use_checkpoint = args.cont
    return n, m, hidden, activation, initializer, optimizer, kwargs, out, \
           use_checkpoint


# saves checkpoint and outputs current step/loss/mu to files
def log(step, loss, params, mu=None):
    global log_prev_time, log_first_time

    now = time.time()
    if log_prev_time and now - log_prev_time < LOG_INTERVAL_IN_SEC:
        return
    if not log_prev_time:
        log_prev_time, log_first_time = now, now
    secs_from_start = int(now - log_first_time) + time_delta
    step += step_delta
    message = f'{step} {secs_from_start} {loss}'
    message += f' {mu}' if mu else ''
    print(message)
    with open(out_file, 'a') as file:
        file.write(message + '\n')
    pickle.dump((step, secs_from_start, params),
                open(out_file + '.ckpt', "wb"))
    log_prev_time = now


# generates m random points from n-dim Rosenbrock
# in case x_norm/y_norm are not provided, normalizes dividing by max
def get_rand_rosenbrock_points(n, m, x_range, x_norm=None, y_norm=None):
    np.random.seed(SEED)
    x = np.random.uniform(*x_range, (m, n))
    y = np.zeros(shape=(m, 1), dtype=NP_DATA_TYPE)
    # include local and global minima for 4 <= N <= 7
    x[0, :] = ([-1] + [1] * (n - 1))
    x[1, :] = [1] * n
    for s in range(m):
        for i in range(n - 1):
            y[s] += 100 * ((x[s, i + 1] - x[s, i] ** 2) ** 2) + (1 - x[s, i]) ** 2
    # do normalization
    x_norm = x_norm or max(x_range)
    y_norm = y_norm or max(y)[0]
    x = x / x_norm
    y = y / y_norm - 0.5
    return x, y, x_norm, y_norm


# calculates Jacobian matrix for y with respect to x
def jacobian(y, x):
    m = y.shape[0]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(TF_DATA_TYPE, size=m),
    ]
    _, j = tf.while_loop(
        lambda i, _: i < m,
        lambda i, res: (i + 1, res.write(i, tf.gradients(y[i], x)[0])),
        loop_vars)
    return j.stack()


# performs network training and updates params values according to
# Levenberg-Marquardt optimization.
def train_lm(feed_dict, params, y_hat, r, loss, mu, mu_inc, mu_dec, max_inc):
    neurons_cnt = params.shape[0].value

    mu_current = tf.placeholder(TF_DATA_TYPE, shape=[1])
    i = tf.eye(neurons_cnt, dtype=TF_DATA_TYPE)
    y_hat_flat = tf.squeeze(y_hat)
    j = jacobian(y_hat_flat, params)
    j_t = tf.transpose(j)
    hess = tf.matmul(j_t, j)
    g = tf.matmul(j_t, r)
    p_store = tf.Variable(tf.zeros([neurons_cnt], dtype=TF_DATA_TYPE))
    hess_store = tf.Variable(tf.zeros((neurons_cnt, neurons_cnt), dtype=TF_DATA_TYPE))
    g_store = tf.Variable(tf.zeros((neurons_cnt, 1), dtype=TF_DATA_TYPE))
    save_params = tf.assign(p_store, params)
    restore_params = tf.assign(params, p_store)
    save_hess_g = [tf.assign(hess_store, hess), tf.assign(g_store, g)]
    dx = tf.matmul(tf.linalg.inv(hess_store + tf.multiply(mu_current, i)),
                   g_store)
    dx = tf.squeeze(dx)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1)
    lm = opt.apply_gradients([(-dx, params)])

    feed_dict[mu_current] = np.array([mu])
    session = tf.Session()
    step = 0
    session.run(tf.global_variables_initializer())
    current_loss = session.run(loss, feed_dict)
    while True:
        step += 1
        log(step, current_loss, session.run(params), feed_dict[mu_current][0])
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


def train_tf(feed_dict, params, loss, train_step):
    step = 0
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # calc initial loss
    current_loss = session.run(loss, feed_dict)
    while True:
        step += 1
        log(step, current_loss, session.run(params))
        session.run(train_step, feed_dict)
        current_loss = session.run(loss, feed_dict)


def build_mlp_structure(n, mlp_hidden_structure):
    mlp_structure = [n] + mlp_hidden_structure + [1]
    wb_shapes = []
    for i in range(len(mlp_hidden_structure) + 1):
        wb_shapes.append((mlp_structure[i], mlp_structure[i + 1]))
        wb_shapes.append((1, mlp_structure[i + 1]))
    wb_sizes = [h * w for h, w in wb_shapes]
    neurons_cnt = sum(wb_sizes)
    print(f'Total number of trainable parameters is {neurons_cnt}')
    return neurons_cnt, wb_shapes, wb_sizes


def main():
    global out_file, step_delta, time_delta
    n, m, hidden, activation, initializer, \
        optimizer, kwargs, out_file, use_checkpoint = \
        parse_arguments()

    dp_x, dp_y, x_norm, y_norm = get_rand_rosenbrock_points(n, m, (-2, 2))
    neurons_cnt, wb_shapes, wb_sizes = build_mlp_structure(n, hidden)

    ckpt_data = None
    if use_checkpoint and os.path.exists(out_file + '.ckpt'):
        step_delta, time_delta, ckpt_data = pickle.load(open(out_file + '.ckpt', "rb"))
    else:
        with open(out_file, "a") as file:
            file.write(f'{" ".join(sys.argv[1:])}\n')

    loss, params, r, x, y, y_hat = \
        build_tf_nn(n, m, hidden, activation, initializer, neurons_cnt,
                    wb_shapes, wb_sizes, ckpt_data)

    feed_dict = {x: dp_x, y: dp_y}
    if optimizer == 'lm':
        train_lm(feed_dict, params, y_hat, r, loss, **kwargs)
    else:
        train_step = TF_OPTIMIZERS[optimizer](**kwargs).minimize(loss)
        train_tf(feed_dict, params, loss, train_step)


def build_tf_nn(n, m, hidden, activation, initializer, neurons_cnt, wb_shapes, wb_sizes, ckpt_data):
    # placeholder variables (we have m data points)
    x = tf.placeholder(TF_DATA_TYPE, shape=[m, n])
    y = tf.placeholder(TF_DATA_TYPE, shape=[m, 1])
    if ckpt_data is not None:
        params = tf.Variable(ckpt_data, dtype=TF_DATA_TYPE)
    else:
        params = tf.Variable(initializer([neurons_cnt], dtype=TF_DATA_TYPE))
    tensors = tf.split(params, wb_sizes, 0)
    for i in range(len(tensors)):
        tensors[i] = tf.reshape(tensors[i], wb_shapes[i])
    ws = tensors[0:][::2]
    bs = tensors[1:][::2]
    y_hat = x
    for i in range(len(hidden)):
        y_hat = activation(tf.matmul(y_hat, ws[i]) + bs[i])
    y_hat = tf.matmul(y_hat, ws[-1]) + bs[-1]
    r = y - y_hat
    loss = tf.reduce_mean(tf.square(r))
    return loss, params, r, x, y, y_hat


if __name__ == '__main__':
    main()
