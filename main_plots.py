#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from plots_func import (
    plot_dirichlet_distribution,
    plot_stratification_results,
    plot_algorithm_comparison
)

def load_results(args):
    """Load training results from PKL files."""
    results = {}
    methods = {
        'ours': 'Stratified',
        'comp_grads': 'Compressed Gradients',
        'dp_comp_grads': 'DP + Compressed'
    }
    
    for method_key, method_name in methods.items():
        acc_file = f"saved_exp_info/acc/MNIST_dir_{args.alpha}_{method_key}_p{args.q}_lr0.01_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s10_d1.0_m{args.mu}_s0.pkl"
        if os.path.exists(acc_file):
            with open(acc_file, 'rb') as f:
                data = pickle.load(f)
                # Assuming data contains 'train_loss' and 'test_acc'
                results[method_name] = {
                    'train_loss': data['train_loss'] if 'train_loss' in data else [],
                    'test_acc': data['test_acc'] if 'test_acc' in data else []
                }
                print(f"Loaded results for {method_name}")
        else:
            print(f"File not found: {acc_file}")
    return results

def load_partition_data(args):
    """Load data partition information."""
    partition_file = f"dataset/data_partition_result/MNIST_dir_{args.alpha}.pkl"
    if os.path.exists(partition_file):
        with open(partition_file, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded partition data: {type(data)}")
            
            # Initialize lists
            dataset_sizes = []
            strata_assignments = []
            
            # Handle different possible data structures
            if isinstance(data, dict):
                print(f"Number of clients in partition data: {len(data)}")
                # If data is a dictionary of client data
                for client_id in sorted(data.keys()):
                    client_data = data[client_id]
                    if isinstance(client_data, dict):
                        # If client data is a dictionary with metadata
                        if 'n_train_samples' in client_data:
                            dataset_sizes.append(client_data['n_train_samples'])
                        elif 'samples' in client_data:
                            dataset_sizes.append(len(client_data['samples']))
                        
                        if 'stratum' in client_data:
                            strata_assignments.append(client_data['stratum'])
                    elif isinstance(client_data, (list, np.ndarray)):
                        # If client data is directly the samples
                        dataset_sizes.append(len(client_data))
            
            # Calculate strata sizes from assignments
            if strata_assignments:
                unique_strata = sorted(set(strata_assignments))
                strata_sizes = [strata_assignments.count(s) for s in unique_strata]
            else:
                strata_sizes = []
            
            print(f"Dataset sizes: min={min(dataset_sizes) if dataset_sizes else 'N/A'}, "
                  f"max={max(dataset_sizes) if dataset_sizes else 'N/A'}")
            print(f"Number of strata: {len(strata_sizes)}")
            print(f"Strata sizes: {strata_sizes}")
            
            return np.array(dataset_sizes), strata_sizes, strata_assignments
    else:
        print(f"Partition file not found: {partition_file}")
        return np.array([]), [], []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', type=str, default='all', choices=['all', 'dirichlet', 'stratification', 'comparison'])
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--n_SGD', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_iter', type=int, default=99)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--K_desired', type=int, default=2048)
    parser.add_argument('--d_prime', type=int, default=9)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--dp_alpha', type=float, default=0.1616)
    
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load partition data once for both dirichlet and stratification plots
    dataset_sizes, strata_sizes, strata_assignments = load_partition_data(args)
    
    if args.plot_type in ['dirichlet', 'all']:
        print(f"Generating Dirichlet distribution plots for Non-IID α={args.alpha}...")
        plot_dirichlet_distribution(args.alpha, dataset_sizes, args.dataset)
    
    if args.plot_type in ['stratification', 'all']:
        print(f"Generating stratification results plots for Non-IID α={args.alpha}...")
        plot_stratification_results(strata_sizes, strata_assignments, args.alpha, args.dataset)
    
    if args.plot_type in ['comparison', 'all']:
        print(f"Generating algorithm comparison plots for Non-IID α={args.alpha}, q={args.q}...")
        results = load_results(args)
        if results:
            plot_algorithm_comparison(results, args.alpha, args.q, args.dataset)
        else:
            print("No results found. Please make sure the result files exist.")

if __name__ == "__main__":
    main() 