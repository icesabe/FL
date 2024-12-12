#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import config
from utils import *
from copy import deepcopy
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

if config.USE_GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# turn data into Variable, do .cuda() when USE_GPU is True
def get_variable(x):
    x = Variable(x)
    return x.cuda() if config.USE_GPU else x
    # requires_grad=True with tensor x in newer PyTorch versions

def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels)

def client_compress_gradient(client_model, train_data, d_prime):
    """
    Compute and compress gradients for a client
    """
    # Get gradient from all batches
    accumulated_grad = None
    batch_count = 0
    
    for features, labels in train_data:
        if config.USE_GPU:
            features = features.cuda()
            labels = labels.cuda()
            
        predictions = client_model(features)
        loss = loss_classifier(predictions, labels)
        loss.backward()
        
        # Accumulate gradients
        if accumulated_grad is None:
            accumulated_grad = []
            for param in client_model.parameters():
                if param.grad is not None:
                    accumulated_grad.append(param.grad.data.clone())
        else:
            for i, param in enumerate(client_model.parameters()):
                if param.grad is not None:
                    accumulated_grad[i] += param.grad.data
        
        batch_count += 1
        client_model.zero_grad()
        
    # Average the accumulated gradients
    for grad in accumulated_grad:
        grad /= batch_count
        
    # Flatten averaged gradient
    grad = []
    for acc_grad in accumulated_grad:
        grad.append(acc_grad.flatten())
    flat_grad = torch.cat(grad)
    
    # Compress using k-means
    grad_np = flat_grad.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=d_prime, random_state=0)
    indices = kmeans.fit_predict(grad_np.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    
    return centers, indices

def collect_compressed_gradients(model, training_sets, d_prime):
    """
    Collect compressed gradients from all clients
    Args:
        model: global model
        training_sets: list of training datasets
        d_prime: compression parameter
    Returns:
        all_compressed_grads: compressed gradients from all clients
        all_indices: indices for each client's compressed gradients
    """
    all_compressed_grads = []
    all_indices = []
    
    for client_id, train_data in enumerate(training_sets):
        # Each client computes and compresses their gradient
        local_model = deepcopy(model)
        compressed_grad, indices = client_compress_gradient(local_model, train_data, d_prime)
        
        # Server collects compressed gradients
        all_compressed_grads.append(compressed_grad)
        all_indices.append(indices)
    
    return np.array(all_compressed_grads), all_indices

def stratify_clients_compressed_gradients(args, compressed_grads):
    """
    Args:
        args: Arguments
        compressed_grads: Compressed gradients from clients
    """
    # Uses compressed gradients directly - no need for PCA
    data = compressed_grads
    print("Shape of compressed gradients:", data.shape)

    # Prototype Based Clustering: KMeans
    model = KMeans(n_clusters=args.strata_num)
    model.fit(data)
    pred_y = model.predict(data)
    pred_y = list(pred_y)
    result = []
    
    # put indexes into result
    for num in range(args.strata_num):
        one_type = []
        for index, value in enumerate(pred_y):
            if value == num:
                one_type.append(index)
        result.append(one_type)
    print("Stratification result:", result)
    
    save_path = f'dataset/stratify_result/{args.dataset}_{args.partition}.pkl'
    with open(save_path, 'wb') as output:
        pickle.dump(result, output)

    # print silhouette_score
    s_score = metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean')
    print("strata_num：", args.strata_num, " silhouette_score：", s_score, "\n")
    
    return result

def accuracy_dataset(model, dataset):
    """Compute the accuracy {}% of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy

def loss_dataset(model, train_data, loss_classifier):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        loss += loss_classifier(predictions, labels)

    loss /= idx + 1 #average loss, idx is batch index
    return loss

def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_classifier):
    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = get_variable(features)
        labels = get_variable(labels)

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_classifier(predictions, labels)
        
        tensor_1 = list(model.parameters())
        tensor_2 = list(model_0.parameters())
        norm = sum(
            [
                torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                for i in range(len(tensor_1))
            ]
        )
        batch_loss += mu / 2 * norm
        
        batch_loss.backward()
        optimizer.step()

def FedProx_random_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples) #(k,)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))
    
    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)
        
    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [weights[client] for client in sampled_clients]

        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(sum(weights_) * layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_importance_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [1 / n_sampled] * n_sampled
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_stratified_sampling(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    K_desired: int,
    d_prime: int,
):
    """
    Modified FedProx with stratified sampling based on gradient norms and K_desired samples
    """
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # STRATIFY THE CLIENTS
    # 1. Get compressed gradients from all clients
    compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

    # 2. Stratify clients based on compressed gradients
    stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)

    # 3. Calculate allocation numbers based on gradient norms
    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA, args.sample_ratio)
    print("Allocation numbers:", allocation_number)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # 4. Compute sampling probabilities based on gradient norms
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                grad_norm = np.linalg.norm(compressed_grads[k])
                client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12)
        
        # Sampling clients based on stratification
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        selected = []
        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # Create a subset of K_desired samples if the dataset is larger
            total_samples = len(training_sets[k].dataset)
            if K_desired and total_samples > K_desired:
                # Create indices for random sampling without replacement
                indices = torch.randperm(total_samples)[:K_desired]
                subset = torch.utils.data.Subset(training_sets[k].dataset, indices)
                train_loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
            else:
                train_loader = training_sets[k]

            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                train_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        weights_ = [1 / n_sampled] * n_sampled
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_stratified_dp_sampling(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    alpha: float,  # Privacy parameter from FedSampling
    M: int,        # Maximum response value for the Estimator
    K_desired: int, # Desired sample size
    d_prime: int,   
):  
    # Initialize Estimator for privacy-preserving sampling
    train_users = {k: range(len(dl.dataset)) for k, dl in enumerate(training_sets)}
    estimator = Estimator(train_users, alpha, M)

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 1. each client sends compressed gradients **************************************
    # Get compressed gradients from all clients
    compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

    # 2. Stratify clients based on compressed gradients ******************************
    # Use compressed gradients for stratification
    stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

    # 3. Server computes the m_h *****************************************************
    # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
    # Note: S_h is calculated using compressed gradients, not restored gradients
    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA, args.sample_ratio)
    print(allocation_number)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # Estimate the total population size with privacy preservation
        hatN = estimator.estimate()
        print(f"Estimated population size (hatN): {hatN}")

        # Sampling clients based on stratification and privacy-preserving estimates
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            for k in range(N_CLIENTS):
                if k in cls:
                    chosen_p[j][k] = round(1 / SIZE_STRATA[j], 12)
        

        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        selected = []
        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # local data sampling
            sampled_features, sampled_labels = local_data_sampling(
                training_sets[k], 
                K_desired, 
                hatN
            )

            if sampled_features is not None and len(sampled_features) > 0:
               
                sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                sampled_loader = torch.utils.data.DataLoader(
                    sampled_dataset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                sampled_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        # Data-size proportional weights
        #weights_ = [weights[client] for client in selected]
        weights_ = [1/n_sampled]*n_sampled

        for layer_weights in new_model.parameters():
            layer_weights.data.sub_(sum(weights_) * layer_weights.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def calculate_aggregation_weights(stratify_result, chosen_p, selected_clients):
    """
    Calculate aggregation weights according to the formula:
    w_{t+1} = \frac{1}{N} \sum_{h=1}^{H} N_h \frac{1}{m_h} \sum_{k=1}^{m_h} \frac{1}{p_t^k} w_{t+1}^k
    """
    N_h = [len(cls) for cls in stratify_result]  # Size of each stratum
    N = sum(N_h)  # Total number of clients
    
    # Count selected clients in each stratum
    m_h = [0] * len(stratify_result)
    for k in selected_clients:
        for h, cls in enumerate(stratify_result):
            if k in cls:
                m_h[h] += 1
                break
    
    # Calculate weights for each selected client
    weights_ = []
    client_to_stratum = {}
    
    # Create mapping of client to stratum
    for h, cls in enumerate(stratify_result):
        for k in cls:
            client_to_stratum[k] = h
            
    # Calculate weight for each selected client
    for k in selected_clients:
        h = client_to_stratum[k]
        if m_h[h] > 0:  # Avoid division by zero
            weight = (N_h[h] / (N * m_h[h])) * (1 / chosen_p[h][k])
            weights_.append(weight)
        else:
            weights_.append(0)
    
    # Normalize weights
    #weights_sum = sum(weights_)
    #if weights_sum > 0:
    #    weights_ = [w / weights_sum for w in weights_]
    #else:
    #    weights_ = [1.0 / len(selected_clients)] * len(selected_clients)
    
    return weights_, N


def FedProx_stratified_sampling_compressed_gradients(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    K_desired: int,
    d_prime: int,
):  
    """
    Federated learning with stratified sampling using compressed gradients (non-DP version)
    """
    print("Running FedProx with stratified sampling using compressed gradients")

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 1. Get compressed gradients from all clients
    compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

    # 2. Stratify clients based on compressed gradients
    stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)

    # 3. Calculate allocation numbers if needed
    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA, args.sample_ratio)
    print("Allocation numbers:", allocation_number)

    # Initialize history tracking
    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # 4. Compute sampling probabilities based on gradient norms
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                grad_norm = np.linalg.norm(compressed_grads[k])
                client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12)
        
        # Sampling clients based on stratification
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        selected = []
        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            
            # Calculate sampling probability based on actual dataset size
            total_samples = len(training_sets[k].dataset)
            # Ensure sampling probability is valid (between 0 and 1)
            sampling_prob = min(1.0, max(0.0, float(K_desired) / float(total_samples)))
            print(f"Client {k} - Total samples: {total_samples}, K_desired: {K_desired}, Sampling prob: {sampling_prob}")
            
            # Sample data points
            sampled_features = []
            sampled_labels = []
            
            for features, labels in training_sets[k]:
                sample_mask = np.random.binomial(n=1, p=sampling_prob, size=len(features))
                selected_features = features[sample_mask == 1]
                selected_labels = labels[sample_mask == 1]
                
                if len(selected_features) > 0:
                    sampled_features.append(selected_features)
                    sampled_labels.append(selected_labels)

            if sampled_features:
                sampled_features = torch.cat(sampled_features)
                sampled_labels = torch.cat(sampled_labels)
                sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                sampled_loader = torch.utils.data.DataLoader(
                    sampled_dataset,
                    batch_size=args.batch_size,
                    shuffle=True
                )

            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                sampled_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        if not clients_params:  # Skip aggregation if no clients had samples
            print("Warning: No clients had valid samples in this round")
            continue

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        #uniform weights
        #weights_ = [1/len(clients_params)]*len(clients_params)
        # Use data-size proportional weights for each selected client
        #weights_ = [weights[client] for client in selected]
        #weights_ = [1/n_sampled]*n_sampled
        #weights_ = [1 / n_sampled] * n_sampled
        #for layer_weigths in new_model.parameters():
        #   layer_weigths.data.sub_(layer_weigths.data)
        weights_, N = calculate_aggregation_weights(stratify_result, chosen_p, sampled_clients_for_grad)
        weights_sum = sum(weights_)
        print(f"Sum of weights: {weights_sum}")  # Should be close to 1/N * sum over all strata

        #add an assertion (with some tolerance)
        assert abs(weights_sum - (1.0 / N)) < 1e-6, "Weights do not sum to 1/N as expected."
        #assert abs(sum(weights_) - 1.0) < 1e-6, "Weights do not sum to 1"
        # Aggregate model updates
        #for layer_weights in new_model.parameters():
        #    layer_weights.data.sub_(sum(weights_) * layer_weights.data)
        for layer_weights in new_model.parameters():
            layer_weights.data.sub_(layer_weights.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)
       
        #for layer_weights in new_model.parameters():
        #    layer_weights.data.sub_(sum(weights_) * layer_weights.data)

        #for k, client_hist in enumerate(clients_params):
        #    for idx, layer_weights in enumerate(new_model.parameters()):
        #        contribution = client_hist[idx].data * weights_[k]
        #        layer_weights.data.add_(contribution)

        model = new_model

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_stratified_dp_sampling_compressed_gradients(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    alpha: float,  # Privacy parameter from FedSampling
    M: int,        # Maximum response value for the Estimator
    K_desired: int, # Desired sample size
    d_prime: int,  # Compression parameter
):  
    # Initialize Estimator for privacy-preserving sampling
    train_users = {k: range(len(dl.dataset)) for k, dl in enumerate(training_sets)}
    estimator = Estimator(train_users, alpha, M)

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 1. Get compressed gradients from all clients
    compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

    # 2. Stratify clients based on compressed gradients ******************************
    # Use compressed gradients for stratification
    stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

    # 3. Server computes the m_h *****************************************************
    # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
    # Note: S_h is calculated using compressed gradients, not restored gradients
    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA, args.sample_ratio)
    print(allocation_number)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # Estimate the total population size with privacy preservation
        hatN = estimator.estimate()
        print(f"Estimated population size (hatN): {hatN}")

        # 4. Server computes p_t^k ***************************************************
        # Note: ||Z_t^k|| is calculated using compressed gradients, not restored gradients
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                    # Find ||Z_t^k|| of this client and store it in an array
                    grad_norm = np.linalg.norm(compressed_grads[k])
                    client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12)
        
        # Sampling clients based on stratification and privacy-preserving estimates
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        selected = []
        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # local data sampling
            sampled_features, sampled_labels = local_data_sampling(
                training_sets[k], 
                K_desired, 
                hatN
            )

            if sampled_features is not None and len(sampled_features) > 0:
               
                sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                sampled_loader = torch.utils.data.DataLoader(
                    sampled_dataset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                sampled_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        #uniform weights
        #weights_ = [1/n_sampled]*n_sampled
        # Use data-size proportional weights for each selected client
        #weights_ = [weights[client] for client in selected]
        #weights_sum = sum(weights_)
        #weights_ = [w/weights_sum for w in weights_]
        weights_, N = calculate_aggregation_weights(stratify_result, chosen_p, sampled_clients_for_grad)
        #assert abs(sum(weights_) - 1.0) < 1e-6, "Weights do not sum to 1"
        weights_sum = sum(weights_)
        print(f"Sum of weights: {weights_sum}")  # Should be close to 1/N * sum over all strata

        #add an assertion (with some tolerance)
        assert abs(weights_sum - (1.0 / N)) < 1e-6, "Weights do not sum to 1/N as expected."
        # Aggregate model updates
        for layer_weights in new_model.parameters():
            layer_weights.data.sub_(layer_weights.data)
        #for layer_weights in new_model.parameters():
        #    layer_weights.data.sub_(sum(weights_) * layer_weights.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)
        #weights_ = [1 / n_sampled] * n_sampled
        #for layer_weigths in new_model.parameters():
        #   layer_weigths.data.sub_(layer_weigths.data)


        #for layer_weights in new_model.parameters():
        #    layer_weights.data.sub_(sum(weights_) * layer_weights.data)

        #for k, client_hist in enumerate(clients_params):
        #    for idx, layer_weights in enumerate(new_model.parameters()):
        #        contribution = client_hist[idx].data * weights_[k]
        #        layer_weights.data.add_(contribution)

        model = new_model

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        # Decrease the learning rate
        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist



def run(args, model_mnist, n_sampled, list_dls_train, list_dls_test, file_name):
    """RUN FEDAVG WITH RANDOM SAMPLING"""
    if args.sampling == "random" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_random_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH IMPORTANCE SAMPLING"""
    if args.sampling == "importance" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_importance_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH OURS SAMPLING"""
    if (args.sampling == "ours") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_sampling(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.K_desired,
            args.d_prime,
        )
        
    """RUN FEDAVG WITH dp sampling """
    if (args.sampling == "dp") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_dp_sampling(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.alpha,
            args.M,
            args.K_desired,
            )
    """RUN FEDAVG WITH dp sampling and compressed client gradients"""
    if (args.sampling == "comp_grads") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_sampling_compressed_gradients(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.K_desired,
            args.d_prime,
            )
    """RUN FEDAVG WITH dp sampling and compressed client gradients"""
    if (args.sampling == "dp_comp_grads") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_dp_sampling_compressed_gradients(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.alpha,
            args.M,
            args.K_desired,
            args.d_prime,
            )
