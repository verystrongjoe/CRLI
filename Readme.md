# README
It's a pytorch implementation of AAAI2021 paper "Learning Representation for Incomplete Time-series Clustering" by Qianli Ma, Chuxin Chen, and Sen Li.

# To run your own model
```
python main.py --dataset_name xxx
```

## Parameter detail
The results are obtained by running grid search on following parameters:

- lambda_kmeans ∈ {1e-3,1e-6,1e-9}

- G_hiddensize ∈ {50,100,150}

- G_layer ∈ {1,2,3}
