import math

import tensorflow as tf
import numpy as np

from .configs import *


def _create_U_or_V(n, rank, mu, std):
    # _mu = math.sqrt(mu / rank)
    # _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
    _mu = 0
    _std = 1 / rank
    return tf.Variable(tf.truncated_normal([n, rank], _mu, _std))


def init_models(batch_manager):
    m, n = batch_manager.n_user, batch_manager.n_item
    # mu = batch_manager.mu
    K = len(RANKS)

    i = tf.placeholder(tf.int32, [None], name='i')
    j = tf.placeholder(tf.int32, [None], name='j')
    r = tf.placeholder(tf.float32, [None], name='r')

    alpha = tf.Variable(tf.ones([m, K]) / math.sqrt(K), name="alpha")
    beta = tf.Variable(tf.ones([n, K]) / math.sqrt(K), name="beta")

    sigma_2 = tf.Variable(0.1)
    sigma_U_2 = tf.Variable(0.1)
    sigma_V_2 = tf.Variable(0.1)
    b_alpha = tf.Variable(0.1)
    b_beta = tf.Variable(0.1)
    mu_alpha = tf.Variable(0.1)
    mu_beta = tf.Variable(0.1)

    Us, Vs, _r_hats = [], [], []
    for k in RANKS:
        U = _create_U_or_V(m, k, batch_manager.mu, batch_manager.std)
        V = _create_U_or_V(n, k, batch_manager.mu, batch_manager.std)
        U_lookup = tf.nn.embedding_lookup(U, i)
        V_lookup = tf.nn.embedding_lookup(V, j)

        _r_hat = tf.reduce_sum(tf.multiply(U_lookup, V_lookup), 1)
        Us.append(U)
        Vs.append(V)
        _r_hats.append(_r_hat)

    r_hats = tf.stack(_r_hats, axis=1)
    alpha_lookup = tf.nn.embedding_lookup(alpha, i)
    beta_lookup = tf.nn.embedding_lookup(beta, j)

    # print(alpha_lookup, beta_lookup, tf.square(r - r_hats), r - r_hats)
    # print(alpha_lookup * beta_lookup * tf.square(r - r_hats))
    rec_loss = tf.reduce_sum(alpha_lookup * beta_lookup *
                             tf.square(tf.reshape(r, [-1, 1]) - r_hats))
    # print(rec_loss)
    # input('')

    reg_loss = tf.add_n([tf.reduce_sum(tf.square(U))
                         for U in Us]) / (2 * sigma_U_2)
    reg_loss += tf.add_n([tf.reduce_sum(tf.square(V))
                          for V in Vs]) / (2 * sigma_V_2)
    reg_loss += tf.reduce_sum(tf.math.abs(alpha - mu_alpha)) / b_alpha
    reg_loss += tf.reduce_sum(tf.math.abs(beta - mu_beta)) / b_beta
    # reg_loss += tf.dtypes.cast(
    #     tf.size(r), dtype=tf.float32) * tf.math.log(sigma_2) / 2
    # reg_loss += K * m * tf.math.log(sigma_U_2) / 2
    # reg_loss += K * n * tf.math.log(sigma_V_2) / 2
    # reg_loss += K * m * tf.math.log(b_alpha)
    # reg_loss += K * n * tf.math.log(b_beta)

    loss = rec_loss / (2 * sigma_2) + reg_loss
    # print(loss)
    # print(rec_loss)
    # print(reg_loss)
    # input('')

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_ops = [
        optimizer.minimize(loss, var_list=Us),
        # optimizer.minimize(loss, var_list=[alpha]),
        optimizer.minimize(loss, var_list=Vs),
        # optimizer.minimize(loss, var_list=[beta]),
    ]
    assign_ops = [
        tf.assign(sigma_2, rec_loss / tf.dtypes.cast(tf.size(r), tf.float32)),
        tf.assign(sigma_U_2,
                  tf.reduce_mean([tf.reduce_sum(tf.square(U)) / m
                                  for U in Us])),
        tf.assign(sigma_V_2,
                  tf.reduce_mean([tf.reduce_sum(tf.square(V)) / n
                                  for V in Vs])),
        tf.assign(mu_alpha, tf.reduce_mean(alpha)),
        tf.assign(b_alpha, tf.reduce_mean(tf.math.abs(alpha - mu_alpha))),
        tf.assign(mu_beta, tf.reduce_mean(beta)),
        tf.assign(b_beta, tf.reduce_mean(tf.math.abs(beta - mu_beta))),
    ]

    alphabeta = alpha_lookup * beta_lookup
    # alphabeta_normalized = alphabeta / tf.reshape(
    #     tf.reduce_sum(alphabeta, axis=1), [-1, 1])
    alphabeta_normalized = alphabeta
    # print(r_hats)
    # print(alphabeta_normalized)
    # print(r_hats * alphabeta_normalized)
    # input('')
    rmse = tf.reduce_mean(
        tf.square(r - tf.reduce_sum(r_hats * alphabeta_normalized, axis=1)))

    return {
        'i': i,
        'j': j,
        'r': r,
        'train_ops': train_ops,
        'assign_ops': assign_ops,
        'rmse': rmse,
        #
        'loss': loss,
        'mu_alpha': mu_alpha,
        'b_alpha': b_alpha,
    }
