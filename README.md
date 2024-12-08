Google Colab Link: https://colab.research.google.com/drive/1-9u0vMrjcoQcXVXqSAP9ltl-yIz8MQxg <br />
<br />
🟥 As of December 7, 2024, 5:24 PM 🟥 <br />
**FedProx_stratified_dp_sampling_compressed_gradients()** in fedprox_func.py is different from FedProx_stratified_dp_sampling() in the following ways: <br />
1. Uses compressed gradients of each client to stratify the clients. <br />
       - The functions used are client_compress_gradient(), collect_compressed_gradients(), and stratify_clients_compressed_gradients().
2. Uses Neyman allocation with N_h (number of clients in stratum h) and S_h (variability of stratum h in terms of clients' compressed gradients) to find m_h. <br />
       - The function used is cal_allocation_number_NS().
3. Uses ||(Z_t)^k||, norm of compressed gradients of each client to compute the client's (p_t)^k. <br />
<br />
The code above needs to be double checked. More changes still need to be made so that the code is aligned with the pseudocode Jordan wrote in our meeting today, specifically, the parts he wrote after "Each round:"<br />
<br />
If time permits: <br />
1. Add exception checks to check for division by zero. <br />
2. Reorganize functions (put some functions in fedprox_func.py to utils.py). <br />
3. Use recovered or restored gradients instead of compressed gradients in using Neyman allocation to find m_h and (p_t)^k.


<br />
<br />
<br />

# FedSTS: A Stratified Client Selection Framework for Consistently Fast Federated Learning

A PyTorch implementation of our paper FedSTS: A Stratified Client Selection Framework for Consistently Fast Federated Learning.

## Dependencies
+ Python (>=3.6)
+ PyTorch (>=1.7.1)
+ NumPy (>=1.19.2)
+ Scikit-Learn (>=0.24.1)
+ Scipy (>=1.6.1)

To install all dependencies:
```
pip install -r requirements.txt
```

## Running an experiment

Here we provide the implementation of Stratified Client Selection Scheme along with MNIST, FMNIST and CIFAR-10 dataset. This code takes as input:

- The `dataset` used.
- The data `partition` method used. partition ∈ { iid, dir_{alpha}, shard }
- The `sampling` scheme used. sampling ∈ { random, importance, ours }
- The percentage of clients sampled `sample_ratio`. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.
- The learning rate `lr` used.
- The batch size `batch_size` used.
- The number of SGD run locally `n_SGD` used.
- The number of rounds of training `n_iter`.
- The number of strata `strata_num` used in ours sampling.
- The learning rate `decay` used after each SGD. We consider no decay in our experiments, decay=1.
- The local loss function regularization parameter `mu`. FedProx with µ = 0 and without systems heterogeneity (no stragglers) corresponds to FedAvg.
- The `seed` used to initialize the training model. We use 0 in all our experiments.
- Force a boolean equal `force` to True when a simulation has already been run but needs to be rerun.
- The privacy parameter `alpha` used in DP sampling (default=0.5)
- The maximum response value `M` for the Estimator in DP sampling (default=300)
- The desired sample size `K_desired` for local data sampling (default=2048)

+ To train and evaluate on MNIST:
```


+ To train and evaluate on MNIST:
```
python main_mnist.py --dataset=MNIST --partition=iid --sampling=random --sample_ratio=0.1 --lr=0.01 --batch_size=50 --n_SGD=50 --n_iter=200 --strata_num=10 --decay=1.0 --mu=0.0 --seed=0 --force=False --alpha=0.5 --M=300 --K_desired=2048
```

+ To train and evaluate on FMNIST:
```
python main_fmnist.py --dataset=FMNIST \
    --partition=shard \
    --sampling=importance \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```

+ To train and evaluate on CIFAR-10:
```
python main_cifar10.py --dataset=CIFAR10 \
    --partition=dir_0.001 \
    --sampling=ours \
    --sample_ratio=0.1 \
    --lr=0.05 \
    --batch_size=50 \
    --n_SGD=80 \
    --n_iter=800 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```

+ To train and evaluate with DP sampling on MNIST:
```
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=dp \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=50 \
    --n_SGD=50 \
    --n_iter=200 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=False
```

Every experiment saves by default the training loss, the testing accuracy, and the sampled clients at every iteration in the folder `saved_exp_info`. The global model and local models histories can also be saved.

## Citation
If you use our code in your research, please cite the following article:
```

```
