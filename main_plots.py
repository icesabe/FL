#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(args):
    """Dynamically load and aggregate training results for accuracy and loss."""
    results = {}
    methods = {
        'random': 'Random',
        'importance': 'Importance',
        'ours': 'Stratified',
        'dp': 'DP',
        'comp_grads': 'Compressed Gradients',
        'dp_comp_grads': 'DP + Compressed'
    }
    
    for method_key, method_name in methods.items():
        # Patterns for accuracy and loss files
        acc_pattern = f"saved_exp_info/acc/MNIST_{args.partition}_{method_key}_p{args.sample_ratio}_lr*_b{args.batch_size}_n*_i*_s*_d*_m*_s*.pkl"
        loss_pattern = f"saved_exp_info/loss/MNIST_{args.partition}_{method_key}_p{args.sample_ratio}_lr*_b{args.batch_size}_n*_i*_s*_d*_m*_s*.pkl"

        acc_files = glob.glob(acc_pattern)
        loss_files = glob.glob(loss_pattern)

        if acc_files and loss_files:
            # Pick the first matching file (or the most recent)
            acc_file = sorted(acc_files, key=os.path.getmtime)[-1]
            loss_file = sorted(loss_files, key=os.path.getmtime)[-1]

            try:
                with open(acc_file, 'rb') as acc_f, open(loss_file, 'rb') as loss_f:
                    acc_data = pickle.load(acc_f)
                    loss_data = pickle.load(loss_f)

                    # Validate data format
                    if isinstance(acc_data, np.ndarray) and isinstance(loss_data, np.ndarray):
                        results[method_name] = {
                            'train_loss': np.mean(loss_data, axis=1).tolist(),  # Aggregate across clients
                            'test_acc': np.mean(acc_data, axis=1).tolist()
                        }
                        print(f"Loaded and aggregated results for {method_name}")
                    else:
                        print(f"Invalid data format in files for {method_name}")
            except Exception as e:
                print(f"Error loading files for {method_name}: {str(e)}")
        else:
            print(f"No files found for method {method_name}")
    
    return results


def plot_algorithm_comparison(results, partition, sample_ratio, dataset='MNIST'):
    """Plot algorithm comparison for training loss and accuracy."""
    if not results:
        print("No results to plot. Ensure valid files are present.")
        return

    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        if 'train_loss' in data and len(data['train_loss']) > 0:
            plt.plot(data['train_loss'], '-', linewidth=2, label=method)
    plt.title(f'Training Loss ({dataset}, Partition={partition}, q={sample_ratio})')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        if 'test_acc' in data and len(data['test_acc']) > 0:
            plt.plot(data['test_acc'], '-', linewidth=2, label=method)
    plt.title(f'Test Accuracy ({dataset}, Partition={partition}, q={sample_ratio})')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plot_path = f'plots/{dataset}_{partition}_comparison_q{sample_ratio}.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', type=str, default='comparison', choices=['comparison'])
    parser.add_argument('--partition', type=str, default='iid', choices=['iid', 'dirichlet', 'shard'])
    parser.add_argument('--sample_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='MNIST')
    
    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Load results and plot
    print(f"Generating algorithm comparison plots for partition={args.partition}, q={args.sample_ratio}...")
    results = load_results(args)
    plot_algorithm_comparison(results, args.partition, args.sample_ratio, dataset=args.dataset)


if __name__ == "__main__":
    main()


