import os

import numpy as np


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


class DatasetManager:
    KIND_MOVIELENS_100K = 'movielens-100k'
    KIND_MOVIELENS_1M = 'movielens-1m'
    KIND_MOVIELENS_10M = 'movielens-10m'
    KIND_MOVIELENS_20M = 'movielens-20m'
    KIND_NETFLIX = 'netflix'

    KIND_OBJECTS = ( \
        (KIND_MOVIELENS_100K, 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'), \
        (KIND_MOVIELENS_1M,  'http://files.grouplens.org/datasets/movielens/ml-1m.zip'), \
        (KIND_MOVIELENS_10M, 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'), \
        (KIND_MOVIELENS_20M, 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'), \
        (KIND_NETFLIX, None)
    )

    def _set_kind_and_url(self, kind):
        self.kind = kind
        for k, url in self.KIND_OBJECTS:
            if k == kind:
                self.url = url
                return True
        raise NotImplementedError()

    def _download_data_if_not_exists(self):
        if not os.path.exists('data/{}'.format(self.kind)):
            os.system('wget {url} -O data/{kind}.zip'.format(
                url=self.url, kind=self.kind))
            os.system(
                'unzip data/{kind}.zip -d data/{kind}/'.format(kind=self.kind))

    def __init_data(self, detail_path, delimiter, header=False):
        current_u = 0
        u_dict = {}
        current_i = 0
        i_dict = {}

        data = []
        with open('data/{}{}'.format(self.kind, detail_path), 'r') as f:
            if header:
                f.readline()

            for line in f:
                cols = line.strip().split(delimiter)
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id, None)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id, None)
                if i is None:
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1

                data.append((u, i, r, t))
            f.close()

        data = np.array(data)
        np.save('data/{}/data.npy'.format(self.kind), data)
        print(data)

    def _init_data(self):
        if self.kind == self.KIND_MOVIELENS_100K:
            self.__init_data('/ml-100k/u.data', '\t')
        elif self.kind == self.KIND_MOVIELENS_1M:
            self.__init_data('/ml-1m/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_10M:
            self.__init_data('/ml-10M100K/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_20M:
            self.__init_data('/ml-20m/ratings.csv', ',', header=True)
        else:
            raise NotImplementedError()

    def _split_data(self):
        data = np.load('data/{}/data.npy'.format(self.kind))
        np.random.shuffle(data)

        n_total = data.shape[0]
        n_train = int(n_total * 0.9)
        n_valid = int(n_train * 0.98)

        np.save('data/{}/train-data.npy'.format(self.kind), data[0:n_valid, :])
        np.save('data/{}/valid-data.npy'.format(self.kind),
                data[n_valid:n_train, :])
        np.save('data/{}/test-data.npy'.format(self.kind), data[n_train:, :])

    def __init__(self, kind):
        _make_dir_if_not_exists('data')
        self._set_kind_and_url(kind)
        self._download_data_if_not_exists()

        if not os.path.exists('data/{}/data.npy'.format(kind)):
            self._init_data()

        if not os.path.exists(
                'data/{}/train-data.npy'.format(kind)) or not os.path.exists(
                    'data/{}/valid-data.npy'.format(kind)
                ) or not os.path.exists('data/{}/test-data.npy'.format(kind)):
            self._split_data()

        self.train_data = np.load('data/{}/train-data.npy'.format(kind))
        self.valid_data = np.load('data/{}/valid-data.npy'.format(kind))
        self.test_data = np.load('data/{}/test-data.npy'.format(kind))

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

    def get_test_data(self):
        return self.test_data


if __name__ == '__main__':
    kind = DatasetManager.KIND_MOVIELENS_100K
    kind = DatasetManager.KIND_MOVIELENS_1M
    kind = DatasetManager.KIND_MOVIELENS_10M
    kind = DatasetManager.KIND_MOVIELENS_20M
    dataset_manager = DatasetManager(kind)
