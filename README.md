# BiasedMF-tensorflow

## Overview

* The tensorflow implementation of BiasedMF(Biased Matrix Factroization) from [Matrix Factorization Techniques for Recommender Systems](http://base.sjtu.edu.cn/~bjshen/2.pdf).
* This repository was forked from [UtsavManiar/Movie_Recommendation_Engine](https://github.com/UtsavManiar/Movie_Recommendation_Engine) and mixed with [JoonyoungYi/NNMF-tensorflow](https://github.com/JoonyoungYi/NNMF-tensorflow).
* I tested MovieLens 100k and 1M dataset.

## Environment

* I've tested this code on Python3.5 and Ubuntu 16.04.
* How to init
```
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
mkdir logs
deactivate
```

* How to run
```
. .venv/bin/activate
python run.py
```


## Performance
* In the `ml-100k` data, rmse is `0.9099072394` in K=7, lambda_value=10.  
* In the `ml-1m` data, rmse is `0.8504005670547485` in K=11, lambda_value=10.
