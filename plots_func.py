import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from numpy.random import dirichlet

def load_pkl(metric, file_name):
    """Load pickle file containing training history"""
    with open(f"saved_exp_info/{metric}/{file_name}.pkl", "rb") as f:
        return pickle.load(f)

def smooth_curve(points, factor=0.8):
    """Smooth the curve using exponential moving average"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def weights_clients(dataset: str):
    """Return normalized weights for clients based on their dataset sizes"""
    if dataset[:5] == "CIFAR":
        dataset = list(dataset)
        dataset[5] = "1"
        dataset = "".join(dataset)

    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pickle.load(output)
    weights = weights / np.sum(weights)
    return weights

def plot_dirichlet_distribution(alpha, n_classes=10, n_clients=100, dataset="CIFAR10"):
    """Plot Dirichlet distribution effects for a specific alpha value"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate Dirichlet distribution
    data = dirichlet([alpha] * n_classes, size=n_clients)
    
    # Plot percentage distribution (column a)
    im = ax1.imshow(data.T, aspect='auto', cmap='YlOrRd')
    ax1.set_xlabel('Clients')
    ax1.set_ylabel('Classes')
    ax1.set_title(f'{dataset} Class Distribution (α = {alpha})')
    plt.colorbar(im, ax=ax1, label='Percentage')
    
    # Plot total samples per class (column b)
    total_samples = np.sum(data, axis=0) * 500  # Assuming 500 samples per client on average
    ax2.bar(range(n_classes), total_samples)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Total Samples')
    ax2.set_title(f'{dataset} Samples per Class (α = {alpha})')
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_dirichlet_distribution_alpha_{alpha}.pdf')
    plt.close()

def plot_stratification_results(alpha, n_strata=10, n_clients=100, dataset="CIFAR10"):
    """Plot stratification results for a specific alpha value"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    file_name = f"dataset/stratify_result/{dataset}_dir_{alpha}.pkl"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            strata = pickle.load(f)
        
        # Plot strata sizes (column a)
        strata_sizes = [len(s) for s in strata]
        ax1.bar(range(len(strata_sizes)), strata_sizes)
        ax1.set_xlabel('Strata')
        ax1.set_ylabel('Number of Clients')
        ax1.set_title(f'{dataset} Strata Sizes (α = {alpha})')
        
        # Plot client distribution (column b)
        client_matrix = np.zeros((n_clients, n_strata))
        for strata_idx, clients in enumerate(strata):
            for client in clients:
                client_matrix[client, strata_idx] = 1
        im = ax2.imshow(client_matrix, aspect='auto', cmap='binary')
        ax2.set_xlabel('Strata')
        ax2.set_ylabel('Clients')
        ax2.set_title(f'{dataset} Client Distribution (α = {alpha})')
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_stratification_results_alpha_{alpha}.pdf')
    plt.close()

def plot_training_metrics(methods, labels, n_SGD, batch_size, n_iter, q, mu, alpha, smooth=True, dataset="CIFAR10", K_desired=2048, d_prime=9, M=100, dp_alpha=0.1616):
    """Plot training metrics (accuracy and loss) for multiple methods with specific sampling ratio"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['red', 'blue', 'green']
    
    # Plot accuracy
    for method, label, color in zip(methods, labels, colors):
        # Construct file name based on method and parameters
        if method == 'dp_comp_grads':
            file_name = f"{dataset}_dir_{alpha}_{method}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0_K{K_desired}_d{d_prime}_a{dp_alpha}_M{M}"
        else:
            file_name = f"{dataset}_dir_{alpha}_{method}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0_K{K_desired}_d{d_prime}"
        
        if os.path.exists(f"saved_exp_info/acc/{file_name}.pkl"):
            history = load_pkl('acc', file_name)
            mean_history = np.mean(history, axis=1)
            if smooth:
                mean_history = smooth_curve(mean_history)
            ax1.plot(mean_history, label=f'{label} (q={q})', color=color)
    
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{dataset} Test Accuracy (Non-IID α = {alpha})')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    for method, label, color in zip(methods, labels, colors):
        # Construct file name based on method and parameters
        if method == 'dp_comp_grads':
            file_name = f"{dataset}_dir_{alpha}_{method}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0_K{K_desired}_d{d_prime}_a{dp_alpha}_M{M}"
        else:
            file_name = f"{dataset}_dir_{alpha}_{method}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0_K{K_desired}_d{d_prime}"
        
        if os.path.exists(f"saved_exp_info/loss/{file_name}.pkl"):
            history = load_pkl('loss', file_name)
            mean_history = np.mean(history, axis=1)
            if smooth:
                mean_history = smooth_curve(mean_history)
            ax2.plot(mean_history, label=f'{label} (q={q})', color=color)
    
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{dataset} Training Loss (Non-IID α = {alpha})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_algorithm_comparison_alpha_{alpha}_q_{q}.pdf')
    plt.close()

def plot_algorithm_comparison(results, alpha, q, dataset='MNIST'):
    """Plot comparison of different algorithms."""
    acc_results, loss_results = results
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    for method, data in loss_results.items():
        plt.plot(range(len(data)), data, label=method, linewidth=2)
    
    plt.title(f'Training Loss (α={alpha}, q={q})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_loss_alpha{alpha}_q{q}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot test accuracy
    plt.figure(figsize=(8, 6))
    for method, data in acc_results.items():
        plt.plot(range(len(data)), data, label=method, linewidth=2)
    
    plt.title(f'Test Accuracy (α={alpha}, q={q})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_accuracy_alpha{alpha}_q{q}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

def sample_dirichlet(alpha, num_samples=1000):
    """
    Samples points from a symmetric Dirichlet distribution.

    Parameters:
    - alpha: float, concentration parameter.
    - num_samples: int, number of samples to generate.

    Returns:
    - samples: np.ndarray of shape (num_samples, 3)
    """
    alpha_vector = [alpha, alpha, alpha]
    samples = dirichlet.rvs(alpha_vector, size=num_samples)
    return samples

def plot_samples(samples, ax, title):
    """
    Plots a 2D scatter plot of Dirichlet samples.

    Parameters:
    - samples: np.ndarray of shape (num_samples, 3)
    - ax: matplotlib axis to plot on.
    - title: str, title of the plot.
    """
    # Project 3D simplex to 2D using a simple transformation
    x = samples[:, 0]
    y = samples[:, 1]
    ax.scatter(x, y, alpha=0.5, s=10, color='blue')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$X_1$', fontsize=12)
    ax.set_ylabel('$X_2$', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

# Define alpha parameters
alpha_values = [0.01, 0.001]
titles = [r'Dirichlet Samples ($\alpha = 0.01$)',
          r'Dirichlet Samples ($\alpha = 0.001$)']

# Create side-by-side scatter plots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for i, alpha in enumerate(alpha_values):
    samples = sample_dirichlet(alpha, num_samples=5000)
    plot_samples(samples, axes[i], titles[i])

plt.tight_layout()
plt.show()
