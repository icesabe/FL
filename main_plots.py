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

def load_partition_data(args):
    """Load data partition information."""
    partition_file = f"dataset/data_partition_result/{args.dataset}_dir_{args.alpha}.pkl"
    print(f"Attempting to load partition data from: {partition_file}")
    
    try:
        if os.path.exists(partition_file):
            with open(partition_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Successfully loaded partition data")
                print(f"Data type: {type(data)}")
                
                dataset_sizes = []
                strata_assignments = []
                
                if isinstance(data, list):
                    dataset_sizes = [len(client_data) for client_data in data]
                elif isinstance(data, dict):
                    dataset_sizes = [len(data[client_id]) for client_id in sorted(data.keys())]
                
                print(f"Number of clients: {len(dataset_sizes)}")
                return np.array(dataset_sizes), [], []
        else:
            print(f"Partition file not found at: {partition_file}")
            return np.array([]), [], []
    except Exception as e:
        print(f"Error loading partition data: {str(e)}")
        return np.array([]), [], []

def load_results(args):
    """Load training results from PKL files."""
    acc_results = {}
    loss_results = {}
    methods = {
        'ours': 'Stratified',
        'comp_grads': 'Compressed Gradients',
        'dp_comp_grads': 'DP + Compressed'
    }
    
    print("\nAttempting to load results...")
    for method_key, method_name in methods.items():
        # Load accuracy data
        acc_file = f"saved_exp_info/acc/{args.dataset}_dir_{args.alpha}_{method_key}_p{args.q}_lr0.01_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s10_d1.0_m{args.mu}_s0.pkl"
        # Load loss data
        loss_file = f"saved_exp_info/loss/{args.dataset}_dir_{args.alpha}_{method_key}_p{args.q}_lr0.01_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s10_d1.0_m{args.mu}_s0.pkl"
        
        try:
            # Load accuracy data
            if os.path.exists(acc_file):
                with open(acc_file, 'rb') as f:
                    acc_data = pickle.load(f)
                    if isinstance(acc_data, np.ndarray):
                        acc_results[method_name] = acc_data.tolist()
                        print(f"Loaded accuracy data for {method_name}")
            
            # Load loss data
            if os.path.exists(loss_file):
                with open(loss_file, 'rb') as f:
                    loss_data = pickle.load(f)
                    if isinstance(loss_data, np.ndarray):
                        loss_results[method_name] = loss_data.tolist()
                        print(f"Loaded loss data for {method_name}")
                
        except Exception as e:
            print(f"Error loading {method_name}: {str(e)}")
    
    return acc_results, loss_results

def main():
    parser = argparse.ArgumentParser(description='Plot federated learning results')
    parser.add_argument('--plot_type', type=str, default='all', 
                      choices=['all', 'dirichlet', 'stratification', 'comparison'])
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--n_SGD', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_iter', type=int, default=99)
    parser.add_argument('--mu', type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    print(f"\nStarting plotting process...")
    print(f"Dataset: {args.dataset}")
    print(f"Alpha: {args.alpha}")
    print(f"q: {args.q}")
    
    # Load partition data
    dataset_sizes, strata_sizes, strata_assignments = load_partition_data(args)
    
    if args.plot_type in ['dirichlet', 'all']:
        print(f"\nGenerating Dirichlet distribution plots...")
        if len(dataset_sizes) > 0:
            plot_dirichlet_distribution(args.alpha, dataset_sizes, args.dataset)
        else:
            print("Warning: No dataset sizes available for Dirichlet distribution plot")
    
    if args.plot_type in ['stratification', 'all']:
        print(f"\nGenerating stratification results plots...")
        if len(strata_sizes) > 0:
            plot_stratification_results(strata_sizes, strata_assignments, args.alpha, args.dataset)
        else:
            print("Warning: No stratification data available")
    
    if args.plot_type in ['comparison', 'all']:
        print(f"\nGenerating algorithm comparison plots...")
        results = load_results(args)
        if results[0] or results[1]:  # Check if either acc_results or loss_results is not empty
            plot_algorithm_comparison(results, args.alpha, args.q, args.dataset)
        else:
            print("\nNo results found. Please check:")
            print(f"1. Directory: saved_exp_info/acc/ and saved_exp_info/loss/")
            print(f"2. File pattern: {args.dataset}_dir_{args.alpha}_*_p{args.q}_*.pkl")

if __name__ == "__main__":
    main()

    """
    # For all plots
python main_plots.py --dataset MNIST --alpha 0.01 --q 0.1

# Or for specific plot types
python main_plots.py --plot_type dirichlet --dataset MNIST --alpha 0.01
python main_plots.py --plot_type comparison --dataset MNIST --alpha 0.01 --q 0.1
    """