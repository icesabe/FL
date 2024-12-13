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
    """Plot algorithm comparison for given alpha and q values."""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        if 'train_loss' in data and len(data['train_loss']) > 0:
            plt.plot(data['train_loss'], '-', linewidth=2, label=method)
    plt.title(f'Training Loss (α={alpha}, q={q})')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        if 'test_acc' in data and len(data['test_acc']) > 0:
            plt.plot(data['test_acc'], '-', linewidth=2, label=method)
    plt.title(f'Test Accuracy (α={alpha}, q={q})')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_comparison_alpha{alpha}_q{q}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_dirichlet_distribution(alpha, dataset_sizes, dataset='MNIST'):
    """Plot Dirichlet distribution visualization."""
    if len(dataset_sizes) == 0:
        print("Warning: No dataset sizes provided for Dirichlet distribution plot")
        return
        
    plt.figure(figsize=(12, 6))
    x = range(len(dataset_sizes))
    plt.bar(x, dataset_sizes, alpha=0.7, label='Client Data Size')
    plt.title(f'Dirichlet Distribution (α={alpha})')
    plt.xlabel('Client Index')
    plt.ylabel('Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics as text
    stats_text = f'Mean: {np.mean(dataset_sizes):.1f}\nStd: {np.std(dataset_sizes):.1f}\nMin: {np.min(dataset_sizes):.1f}\nMax: {np.max(dataset_sizes):.1f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_dirichlet_alpha{alpha}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_stratification_results(strata_sizes, strata_assignments, alpha, dataset='MNIST'):
    """Plot stratification results."""
    if not strata_sizes or not strata_assignments:
        print("Warning: No stratification data provided")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Plot stratum sizes
    plt.subplot(1, 2, 1)
    x = range(len(strata_sizes))
    plt.bar(x, strata_sizes, alpha=0.7, label='Stratum Size')
    plt.title(f'Stratum Sizes (α={alpha})')
    plt.xlabel('Stratum Index')
    plt.ylabel('Number of Clients')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add size labels on top of bars
    for i, v in enumerate(strata_sizes):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    # Plot client assignments
    plt.subplot(1, 2, 2)
    unique_strata = sorted(set(strata_assignments))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_strata)))
    
    # Create scatter plots with labels
    for i, stratum in enumerate(unique_strata):
        mask = np.array(strata_assignments) == stratum
        if np.any(mask):
            plt.scatter(np.where(mask)[0], [1]*np.sum(mask), 
                       c=[colors[i]], 
                       label=f'Stratum {stratum}',
                       alpha=0.7, s=50)
    
    plt.title(f'Client Stratum Assignments (α={alpha})')
    plt.xlabel('Client Index')
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    
    # Adjust legend position and size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., ncol=2 if len(unique_strata) > 10 else 1)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_stratification_alpha{alpha}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Example usage:
if __name__ == "__main__":
    # Plot Dirichlet distribution effects
    alphas = [0.001, 0.01, 0.1, 1.0]
    plot_dirichlet_distribution(alphas)
    
    # Plot stratification results
    plot_stratification_results(alphas)
    
    # Plot algorithm comparison
    plot_algorithm_comparison(
        metric="both",
        n_SGD=80,
        q=0.1,
        mu=0.0,
        alpha=0.001,
        smooth=True
    ) 