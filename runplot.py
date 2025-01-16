import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_pkl(metric, file_name):
    """Load pickle file containing training history."""
    with open(f"saved_exp_info/{metric}/{file_name}.pkl", "rb") as f:
        return pickle.load(f)

def smooth_curve(points, factor=0.8):
    """Smooth the curve using an exponential moving average."""
    smoothed_points = []
    for p in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + p * (1 - factor))
        else:
            smoothed_points.append(p)
    return smoothed_points

# Parameters (adjust as needed)
dataset = "MNIST"
alpha = 0.01
q = 0.1
batch_size = 128
n_SGD = 3
n_iter = 99
mu = 0.0
smooth = True

# Methods and their labels
methods = {
    'ours': 'Stratified',
    'comp_grads': 'Compressed Gradients',
    'dp_comp_grads': 'DP + Compressed'
}

# Create output directory
if not os.path.exists('plots'):
    os.makedirs('plots')

acc_results = {}
loss_results = {}
colors = ['red', 'blue', 'green']  # Assign a color to each method

# Load data
for method_key, method_label in methods.items():
    # Construct file name patterns. If your file naming is different, adjust here:
    if method_key == 'dp_comp_grads':
        acc_file = f"{dataset}_dir_{alpha}_{method_key}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0"
        loss_file = f"{dataset}_dir_{alpha}_{method_key}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0"
    else:
        acc_file = f"{dataset}_dir_{alpha}_{method_key}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0"
        loss_file = f"{dataset}_dir_{alpha}_{method_key}_p{q}_lr0.01_b{batch_size}_n{n_SGD}_i{n_iter}_s10_d1.0_m{mu}_s0"

    # Load accuracy data
    acc_path = f"saved_exp_info/acc/{acc_file}.pkl"
    if os.path.exists(acc_path):
        acc_data = load_pkl('acc', acc_file)
        # Check shape and average if needed
        acc_data = np.array(acc_data)
        # If shape is (num_rounds, num_seeds), average over seeds: axis=1
        # If shape is (num_seeds, num_rounds), average over seeds: axis=0
        # We'll try both patterns:
        if acc_data.ndim == 2:
            # Determine which axis is rounds by choosing the larger dimension as rounds
            if acc_data.shape[0] < acc_data.shape[1]:
                # shape like (num_seeds, num_rounds)
                mean_acc = np.mean(acc_data, axis=0)
            else:
                # shape like (num_rounds, num_seeds)
                mean_acc = np.mean(acc_data, axis=1)
        else:
            # If it's already 1D, just use it directly
            mean_acc = acc_data
        
        if smooth:
            mean_acc = smooth_curve(mean_acc)
        acc_results[method_label] = mean_acc

    # Load loss data
    loss_path = f"saved_exp_info/loss/{loss_file}.pkl"
    if os.path.exists(loss_path):
        loss_data = load_pkl('loss', loss_file)
        loss_data = np.array(loss_data)
        # Average similarly as acc_data
        if loss_data.ndim == 2:
            if loss_data.shape[0] < loss_data.shape[1]:
                mean_loss = np.mean(loss_data, axis=0)
            else:
                mean_loss = np.mean(loss_data, axis=1)
        else:
            mean_loss = loss_data
        
        if smooth:
            mean_loss = smooth_curve(mean_loss)
        loss_results[method_label] = mean_loss

# Plot Accuracy
plt.figure(figsize=(8, 6))
for (method_label, data), c in zip(acc_results.items(), colors):
    plt.plot(data, label=method_label, color=c)
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.title(f'{dataset} Test Accuracy (Non-IID α = {alpha})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/{dataset}_accuracy_alpha_{alpha}_q_{q}.png', dpi=300)
plt.close()

# Plot Loss
plt.figure(figsize=(8, 6))
for (method_label, data), c in zip(loss_results.items(), colors):
    plt.plot(data, label=method_label, color=c)
plt.xlabel('Communication Rounds')
plt.ylabel('Loss')
plt.title(f'{dataset} Training Loss (Non-IID α = {alpha})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/{dataset}_loss_alpha_{alpha}_q_{q}.png', dpi=300)
plt.close()

print("Plots have been saved in the 'plots' directory.")
