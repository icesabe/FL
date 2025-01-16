# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import pickle
# from plots_func import (
#     plot_dirichlet_distribution,
#     plot_stratification_results,
#     plot_algorithm_comparison
# )

# def load_results(args):
#     """Load training results from PKL files."""
#     results = {}
#     methods = {
#         'ours': 'Stratified',
#         'comp_grads': 'Compressed Gradients',
#         'dp_comp_grads': 'DP + Compressed'
#     }
    
#     for method_key, method_name in methods.items():
#         acc_file = f"saved_exp_info/acc/MNIST_dir_{args.alpha}_{method_key}_p{args.q}_lr0.01_b{args.batch_size}_n{args.n_SGD}_i{args.n_iter}_s10_d1.0_m{args.mu}_s0.pkl"
        
#         try:
#             if os.path.exists(acc_file):
#                 with open(acc_file, 'rb') as f:
#                     data = pickle.load(f)
#                     if isinstance(data, dict):
#                         results[method_name] = {
#                             'train_loss': data.get('train_loss', []),
#                             'test_acc': data.get('test_acc', [])
#                         }
#                         print(f"Loaded results for {method_name}")
#                     else:
#                         print(f"Invalid data format for {method_name}")
#             else:
#                 print(f"File not found: {acc_file}")
#         except Exception as e:
#             print(f"Error loading {method_name}: {str(e)}")
            
#     return results

# def load_partition_data(args):
#     """Load data partition information."""
#     partition_file = f"dataset/data_partition_result/MNIST_dir_{args.alpha}.pkl"
#     if os.path.exists(partition_file):
#         with open(partition_file, 'rb') as f:
#             data = pickle.load(f)
#             print(f"Loaded partition data: {type(data)}")
            
#             # Initialize lists
#             dataset_sizes = []
#             strata_assignments = []
            
#             # Handle different possible data structures
#             if isinstance(data, dict):
#                 print(f"Number of clients in partition data: {len(data)}")
#                 # If data is a dictionary of client data
#                 for client_id in sorted(data.keys()):
#                     client_data = data[client_id]
#                     if isinstance(client_data, dict):
#                         # If client data is a dictionary with metadata
#                         if 'n_train_samples' in client_data:
#                             dataset_sizes.append(client_data['n_train_samples'])
#                         elif 'samples' in client_data:
#                             dataset_sizes.append(len(client_data['samples']))
                        
#                         if 'stratum' in client_data:
#                             strata_assignments.append(client_data['stratum'])
#                     elif isinstance(client_data, (list, np.ndarray)):
#                         # If client data is directly the samples
#                         dataset_sizes.append(len(client_data))
            
#             # Calculate strata sizes from assignments
#             if strata_assignments:
#                 unique_strata = sorted(set(strata_assignments))
#                 strata_sizes = [strata_assignments.count(s) for s in unique_strata]
#             else:
#                 strata_sizes = []
            
#             print(f"Dataset sizes: min={min(dataset_sizes) if dataset_sizes else 'N/A'}, "
#                   f"max={max(dataset_sizes) if dataset_sizes else 'N/A'}")
#             print(f"Number of strata: {len(strata_sizes)}")
#             print(f"Strata sizes: {strata_sizes}")
            
#             return np.array(dataset_sizes), strata_sizes, strata_assignments
#     else:
#         print(f"Partition file not found: {partition_file}")
#         return np.array([]), [], []

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--plot_type', type=str, default='all', choices=['all', 'dirichlet', 'stratification', 'comparison'])
#     parser.add_argument('--dataset', type=str, default='MNIST')
#     parser.add_argument('--alpha', type=float, default=0.01)
#     parser.add_argument('--q', type=float, default=0.1)
#     parser.add_argument('--n_SGD', type=int, default=3)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--n_iter', type=int, default=99)
#     parser.add_argument('--mu', type=float, default=0.0)
#     parser.add_argument('--K_desired', type=int, default=2048)
#     parser.add_argument('--d_prime', type=int, default=9)
#     parser.add_argument('--M', type=int, default=100)
#     parser.add_argument('--dp_alpha', type=float, default=0.1616)
    
#     args = parser.parse_args()
    
#     # Create plots directory if it doesn't exist
#     os.makedirs('plots', exist_ok=True)
    
#     # Load partition data once for both dirichlet and stratification plots
#     dataset_sizes, strata_sizes, strata_assignments = load_partition_data(args)
    
#     if args.plot_type in ['dirichlet', 'all']:
#         print(f"Generating Dirichlet distribution plots for Non-IID α={args.alpha}...")
#         plot_dirichlet_distribution(args.alpha, dataset_sizes, args.dataset)
    
#     if args.plot_type in ['stratification', 'all']:
#         print(f"Generating stratification results plots for Non-IID α={args.alpha}...")
#         plot_stratification_results(strata_sizes, strata_assignments, args.alpha, args.dataset)
    
#     if args.plot_type in ['comparison', 'all']:
#         print(f"Generating algorithm comparison plots for Non-IID α={args.alpha}, q={args.q}...")
#         results = load_results(args)
#         if results:
#             plot_algorithm_comparison(results, args.alpha, args.q, args.dataset)
#         else:
#             print("No results found. Please make sure the result files exist.")

# if __name__ == "__main__":
#     main() 

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


