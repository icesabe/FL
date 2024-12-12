#!/bin/bash

# Section 1: Training Experiments
echo "Running training experiments..."

# Run experiments for alpha=0.001
echo "Running experiments for α=0.001..."

# Run experiment with "ours" sampling
python main_mnist.py --dataset=MNIST \
    --partition=dir_0.001 \
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
    --force=True \
    --K_desired=2048 \
    --d_prime=2

# Run experiment with compressed gradients
python main_mnist.py --dataset=MNIST \
    --partition=dir_0.001 \
    --sampling=comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=200 \
    --n_SGD=3 \
    --n_iter=99 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --K_desired=2048 \
    --d_prime=2

# Run experiment with DP and compressed gradients
python main_mnist.py --dataset=MNIST \
    --partition=dir_0.001 \
    --sampling=dp_comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=200 \
    --n_SGD=3 \
    --n_iter=99 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --alpha=0.5 \
    --M=100 \
    --K_desired=2048 \
    --d_prime=2

# Run experiments for alpha=0.01
echo "Running experiments for α=0.01..."

# Repeat the above three experiments with partition=dir_0.01
# [Same commands as above but with --partition=dir_0.01]

# Section 2: Generate Plots
echo "Generating plots..."

# For alpha=0.001
echo "Generating plots for α=0.001..."

# Generate plots for different sampling ratios (q)
for q in 0.1 0.2 0.3 0.5; do
    echo "Generating plots for q=$q..."
    
    # Generate Dirichlet distribution plots
    python main_plots.py --plot_type=dirichlet \
        --alpha=0.001 \
        --q=$q \
        --n_classes=10 \
        --n_clients=100

    # Generate stratification results plots
    python main_plots.py --plot_type=stratification \
        --alpha=0.001 \
        --q=$q \
        --n_strata=10 \
        --n_clients=100

    # Generate algorithm comparison plots
    python main_plots.py --plot_type=comparison \
        --alpha=0.001 \
        --q=$q \
        --n_SGD=80 \
        --mu=0.0 \
        --smooth=True
done

# For alpha=0.01
echo "Generating plots for α=0.01..."

# Repeat the above plotting commands with --alpha=0.01
for q in 0.1 0.2 0.3 0.5; do
    echo "Generating plots for q=$q..."
    
    # Generate Dirichlet distribution plots
    python main_plots.py --plot_type=dirichlet \
        --alpha=0.01 \
        --q=$q \
        --n_classes=10 \
        --n_clients=100

    # Generate stratification results plots
    python main_plots.py --plot_type=stratification \
        --alpha=0.01 \
        --q=$q \
        --n_strata=10 \
        --n_clients=100

    # Generate algorithm comparison plots
    python main_plots.py --plot_type=comparison \
        --alpha=0.01 \
        --q=$q \
        --n_SGD=80 \
        --mu=0.0 \
        --smooth=True
done
######################################################################################
# Or generate all plots at once for a specific combination of parameters
# python main_plots.py --plot_type comparison --dataset MNIST --alpha 0.001 --q 0.1 --n_SGD 3 --batch_size 200 --n_iter 99
######################################################################################
# Usage instructions:
# 1. Make the script executable:
#    chmod +x run.sh
# 2. Run all experiments and generate plots:
#    ./run.sh
# 3. Or run specific sections by commenting out unwanted commands
#
# Available plot types:
# - dirichlet: Shows effect of alpha on client partitioning
# - stratification: Shows stratification results
# - comparison: Compares the three algorithms (accuracy and loss)
# - all: Generates all plots
#
# Parameters:
# --alpha: Dirichlet distribution parameter (0.001 or 0.01)
# --q: Sampling ratio (0.1, 0.2, 0.3, or 0.5)
# --n_classes: Number of classes in the dataset (default: 10)
# --n_clients: Number of clients (default: 100)
# --n_strata: Number of strata for stratification (default: 10)
# --n_SGD: Number of SGD steps (default: 80)
# --mu: FedProx parameter (default: 0.0)
# --smooth: Whether to smooth the curves (default: True)
#
# Output:
# All plots will be saved in the 'plots' directory with names indicating
# the alpha value and sampling ratio (q) used:
# - dirichlet_distribution_alpha_{alpha}.pdf
# - stratification_results_alpha_{alpha}.pdf
# - algorithm_comparison_alpha_{alpha}_q_{q}.pdf