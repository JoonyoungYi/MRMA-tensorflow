import os
import time
import math
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from common.dataset import DatasetManager
from common.batch import BatchManager
from .configs import *
from .models import init_models


def _validate(session, models, batch_manager):
    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['i']: batch_manager.valid_data[:, 0],
            models['j']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2],
        })

    test_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['i']: batch_manager.test_data[:, 0],
            models['j']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2],
        })

    return valid_rmse, test_rmse


def _train(session, kind, models, batch_manager):
    min_valid_rmse = float("Inf")
    min_valid_iter = 0
    final_test_rmse = float('Inf')

    i = batch_manager.train_data[:, 0]
    j = batch_manager.train_data[:, 1]
    r = batch_manager.train_data[:, 2]

    for assign_idx, assign_op in enumerate(models['assign_ops']):
        session.run(
            assign_op,
            feed_dict={
                models['i']: i,
                models['j']: j,
                models['r']: r,
            })

    for iter in range(N_ITER):
        print('\n>> ITER:', iter)

        # for _ in range(100):
        #     results = session.run(
        #         [
        #             models['train_ops'][1], models['_alpha'], models['loss'],
        #             models['rmse']
        #         ],
        #         feed_dict={
        #             models['i']: i,
        #             models['j']: j,
        #             models['r']: r,
        #         })
        #     loss, rmse = results[-2], results[-1]
        #     print(loss, rmse)
        #     print(np.mean(results[-3]))
        #     if math.isnan(loss):
        #         raise Exception("NaN found!")
        # input('')

        for train_op in models['train_ops']:
            for _ in range(100):
                results = session.run(
                    [train_op, models['loss'], models['rmse']],
                    feed_dict={
                        models['i']: i,
                        models['j']: j,
                        models['r']: r,
                    })
                loss, rmse = results[-2], results[-1]
                print(loss, rmse)
                if math.isnan(loss):
                    raise Exception("NaN found!")
            input('')

        for assign_op in models['assign_ops']:
            _, loss, rmse = session.run(
                (assign_op, models['loss'], models['rmse']),
                feed_dict={
                    models['i']: i,
                    models['j']: j,
                    models['r']: r,
                })

            # print(' -', loss, rmse)
            # print(' -', mu_alpha, b_alpha)
            if math.isnan(loss):
                raise Exception("NaN found!")

        valid_rmse, test_rmse = _validate(session, models, batch_manager)
        if valid_rmse < min_valid_rmse:
            min_valid_iter = iter
            min_valid_rmse = valid_rmse
            final_test_rmse = test_rmse

        print('>>', valid_rmse, test_rmse)
        print('>>', min_valid_iter, min_valid_rmse, final_test_rmse)

        if iter > min_valid_iter + N_EARLY_STOP_ITER:
            break


def main(kind):
    batch_manager = BatchManager(kind)
    models = init_models(batch_manager, kind)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())

    _train(session, kind, models, batch_manager)

    session.close()


if __name__ == '__main__':
    # kind = DatasetManager.KIND_MOVIELENS_100K
    kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M

    main(kind)
