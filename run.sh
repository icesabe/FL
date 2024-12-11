#!/bin/bash

# Run experiment with "ours" sampling
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=ours \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True

# Run experiment with compressed gradients
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --K_desired=2048 \
    --d_prime=2

# Run experiment with DP and compressed gradients
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=dp_comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --alpha=0.5 \
    --M=300 \
    --K_desired=2048 \
    --d_prime=2


#chmod +x run.sh
#./run.sh