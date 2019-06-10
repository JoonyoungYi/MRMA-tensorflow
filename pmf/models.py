import math

import tensorflow as tf
import numpy as np

from .configs import *


def _create_U_or_V(n, rank):
    _mu = 0
    _std = 1 / rank
    return tf.Variable(tf.truncated_normal([n, rank], _mu, _std))


def init_models(batch_manager, k=10):
    m, n = batch_manager.n_user, batch_manager.n_item
    # mu = batch_manager.mu

    i = tf.placeholder(tf.int32, [None], name='i')
    j = tf.placeholder(tf.int32, [None], name='j')
    r = tf.placeholder(tf.float32, [None], name='r')

    U = _create_U_or_V(m, k)
    V = _create_U_or_V(n, k)
    U_lookup = tf.nn.embedding_lookup(U, i)
    V_lookup = tf.nn.embedding_lookup(V, j)

    r_hat = tf.reduce_sum(tf.multiply(U_lookup, V_lookup), 1)
    rec_loss = tf.reduce_sum(tf.square(r_hat - r))

    reg_loss = tf.reduce_sum(tf.square(U)) * LAMBDA_U
    reg_loss += tf.reduce_sum(tf.square(V)) * LAMBDA_I
    loss = rec_loss + reg_loss

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss, var_list=[U, V])

    rmse = tf.sqrt(tf.reduce_mean(tf.square(r_hat - r)))

    return {
        'i': i,
        'j': j,
        'r': r,
        'train_op': train_op,
        'loss': loss,
        'rmse': rmse,
        'U': U,
        'V': V,
    }
