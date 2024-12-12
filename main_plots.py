#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from plots_func import (
    plot_dirichlet_distribution,
    plot_stratification_results,
    plot_algorithm_comparison
)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate plots for federated learning experiments')
    
    parser.add_argument('--plot_type', type=str, required=True,
                      choices=['dirichlet', 'stratification', 'comparison', 'all'],
                      help='Type of plot to generate')
    
    # Add dataset parameter
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                      choices=['MNIST', 'CIFAR10'],
                      help='Dataset to plot results for')
    
    # Parameters for all plots
    parser.add_argument('--alpha', type=float, required=True,
                      choices=[0.001, 0.01],
                      help='Alpha value for Dirichlet distribution')
    
    # Parameters for algorithm comparison
    parser.add_argument('--n_SGD', type=int, default=3,
                      help='Number of SGD steps')
    parser.add_argument('--batch_size', type=int, default=200,
                      help='Batch size')
    parser.add_argument('--n_iter', type=int, default=99,
                      help='Number of iterations')
    parser.add_argument('--q', type=float, required=True,
                      choices=[0.1, 0.2, 0.3, 0.5],
                      help='Sampling ratio (q)')
    parser.add_argument('--mu', type=float, default=0.0,
                      help='FedProx parameter mu')
    parser.add_argument('--smooth', type=bool, default=True,
                      help='Whether to smooth the curves')
    
    # Additional parameters
    parser.add_argument('--n_classes', type=int, default=10,
                      help='Number of classes')
    parser.add_argument('--n_clients', type=int, default=100,
                      help='Number of clients')
    parser.add_argument('--n_strata', type=int, default=10,
                      help='Number of strata')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.plot_type in ['dirichlet', 'all']:
        print(f"Generating Dirichlet distribution plots for α={args.alpha}...")
        plot_dirichlet_distribution(
            alpha=args.alpha,
            n_classes=args.n_classes,
            n_clients=args.n_clients,
            dataset=args.dataset
        )
    
    if args.plot_type in ['stratification', 'all']:
        print(f"Generating stratification results plots for α={args.alpha}...")
        plot_stratification_results(
            alpha=args.alpha,
            n_strata=args.n_strata,
            n_clients=args.n_clients,
            dataset=args.dataset
        )
    
    if args.plot_type in ['comparison', 'all']:
        print(f"Generating algorithm comparison plots for α={args.alpha}, q={args.q}...")
        plot_algorithm_comparison(
            metric="both",
            n_SGD=args.n_SGD,
            batch_size=args.batch_size,
            n_iter=args.n_iter,
            q=args.q,
            mu=args.mu,
            alpha=args.alpha,
            smooth=args.smooth,
            dataset=args.dataset
        )

if __name__ == "__main__":
    main() 