import random

from app.trainer import main as train
from app.utils.dataset import DatasetManager

if __name__ == '__main__':
    # kind = DatasetManager.KIND_MOVIELENS_100K
    kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M

    train(kind)
