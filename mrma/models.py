import math

import tensorflow as tf
import numpy as np

from .configs import *


def _create_U_or_V(kind, rank, key):
    U_or_V = np.load('pmf/weights/{}/{}/{}.npy'.format(kind, rank, key))
    return tf.Variable(U_or_V)


def init_models(batch_manager, kind):
    m, n = batch_manager.n_user, batch_manager.n_item
    K = len(RANKS)

    i = tf.placeholder(tf.int32, [None], name='i')
    j = tf.placeholder(tf.int32, [None], name='j')
    r = tf.placeholder(tf.float32, [None], name='r')

    alpha = tf.Variable(tf.ones([m, K]) / math.sqrt(K), name="alpha")
    beta = tf.Variable(tf.ones([n, K]) / math.sqrt(K), name="beta")

    sigma_2 = tf.Variable(0.1)
    sigma_U_2 = tf.Variable(0.1)
    sigma_V_2 = tf.Variable(0.1)
    b_alpha = tf.Variable(1.)
    b_beta = tf.Variable(1.)
    mu_alpha = tf.Variable(1 / math.sqrt(K))
    mu_beta = tf.Variable(1 / math.sqrt(K))

    Us, Vs, _r_hats = [], [], []
    for k in RANKS:
        U = _create_U_or_V(kind, k, 'U')
        V = _create_U_or_V(kind, k, 'V')
        U_lookup = tf.nn.embedding_lookup(U, i)
        V_lookup = tf.nn.embedding_lookup(V, j)

        _r_hat = tf.reduce_sum(tf.multiply(U_lookup, V_lookup), 1)
        Us.append(U)
        Vs.append(V)
        _r_hats.append(_r_hat)

    r_hats = tf.stack(_r_hats, axis=1)
    alpha_lookup = tf.nn.embedding_lookup(alpha, i)
    beta_lookup = tf.nn.embedding_lookup(beta, j)

    rec_loss = tf.reduce_sum(alpha_lookup * beta_lookup *
                             tf.square(tf.reshape(r, [-1, 1]) - r_hats))

    reg_loss = (tf.add_n([tf.reduce_sum(tf.square(U))
                          for U in Us]) / (2 * sigma_U_2))
    reg_loss += (tf.add_n([tf.reduce_sum(tf.square(V))
                           for V in Vs]) / (2 * sigma_V_2))
    reg_loss += (tf.reduce_sum(tf.math.abs(alpha - mu_alpha)) / b_alpha)
    reg_loss += (tf.reduce_sum(tf.math.abs(beta - mu_beta)) / b_beta)
    reg_loss += (tf.dtypes.cast(tf.size(r), dtype=tf.float32) *
                 tf.math.log(sigma_2) / 2)
    reg_loss += (K * m * tf.math.log(sigma_U_2) / 2)
    reg_loss += (K * n * tf.math.log(sigma_V_2) / 2)
    reg_loss += (K * m * tf.math.log(b_alpha))
    reg_loss += (K * n * tf.math.log(b_beta))

    loss = rec_loss / (2 * sigma_2) + reg_loss

    # Optimizer = tf.train.GradientDescentOptimizer
    Optimizer = lambda x: tf.train.AdamOptimizer(x, epsilon=1e-4)
    alpha_optimizer = Optimizer(LEARNING_RATE_ALPHA)
    beta_optimizer = Optimizer(LEARNING_RATE_BETA)
    train_ops = [
        alpha_optimizer.minimize(loss, var_list=Us),
        beta_optimizer.minimize(loss, var_list=[alpha]),
        alpha_optimizer.minimize(loss, var_list=Vs),
        beta_optimizer.minimize(loss, var_list=[beta]),
    ]
    assign_ops = [
        tf.assign(sigma_2, rec_loss / tf.dtypes.cast(tf.size(r), tf.float32)),
        tf.assign(sigma_U_2,
                  tf.reduce_sum([tf.reduce_sum(tf.square(U)) for U in Us])),
        tf.assign(sigma_V_2,
                  tf.reduce_sum([tf.reduce_sum(tf.square(V)) for V in Vs])),
        tf.assign(mu_alpha, tf.reduce_mean(alpha)),
        tf.assign(b_alpha, tf.reduce_mean(tf.math.abs(alpha - mu_alpha))),
        tf.assign(mu_beta, tf.reduce_mean(beta)),
        tf.assign(b_beta, tf.reduce_mean(tf.math.abs(beta - mu_beta))),
    ]

    alphabeta = alpha_lookup * beta_lookup
    alphabeta = alphabeta / tf.reshape(
        tf.reduce_sum(alphabeta, axis=1), [-1, 1])
    r_hat = tf.reduce_sum(r_hats * alphabeta, axis=1)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))

    return {
        'i': i,
        'j': j,
        'r': r,
        'train_ops': train_ops,
        'assign_ops': assign_ops,
        'loss': loss,
        'rmse': rmse,
        #
        '_alpha': alpha,
        # 'mu_alpha': mu_alpha,
        # 'b_alpha': b_alpha,
    }
