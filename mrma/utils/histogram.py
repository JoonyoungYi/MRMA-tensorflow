import os
import math

import matplotlib.pyplot as plt

from ..configs import *


class HistogramManager:
    def __init__(self):
        self.histogram = {}

    def _get_key(self, n, rmse):
        return (n, int(rmse * 20))

    def _add_count(self, n, rmse):
        if not math.isnan(rmse):
            key = self._get_key(n, rmse)
            count = self.histogram.get(key, 0)
            self.histogram[key] = count + 1

    def add_individual_rmse(self, n, individual_rmse):
        for i in range(individual_rmse.shape[0]):
            _rmse = individual_rmse[i]
            self._add_count(n, _rmse)

    def save(self, idx=0):
        # for key, item in sorted(self.histogram.items(), key=lambda x: x[0]):
        #     print('--', key, item)

        rmses = list(i / 20 for i in range(80))  # 0, 0.05, 0.1, 0.15, 0.2, ..., 4.0
        for n in range(MAX_ALPHA_NUM):
            plt.plot(rmses, [
                self.histogram.get(self._get_key(n, rmse), 0) for rmse in rmses
            ])

        model_idx = os.environ['MODEL_IDX']
        if not os.path.exists('figures/{}'.format(model_idx)):
            os.mkdir('figures/{}'.format(model_idx))

        plt.savefig('figures/{}/{}.png'.format(model_idx, idx))
        plt.clf()
