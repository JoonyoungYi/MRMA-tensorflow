# MRMA-tensorflow

## Overview

* The tensorflow implementation of MRMA (Mixture-Rank Matrix Approximation for Collaborative Filtering) from [Mixture-Rank Matrix Approximation for Collaborative Filtering](http://papers.nips.cc/paper/6651-mixture-rank-matrix-approximation-for-collaborative-filtering).
* I tested MovieLens 100k and 1M dataset.

## Environment

* I've tested this code on Python3.5 (tensorflow 1.12) and Ubuntu 16.04.

## Run

* pretrain:
```
python3 -m pmf.trainer
```

* train:
```
python3 -m mrma.trainer
```
