"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

files = [
    (r"D:\STAT561\FL\dataset\MNIST\MNIST_dir_0.01_test_100.pkl", "α=0.01 Test"),
    (r"D:\STAT561\FL\dataset\MNIST\MNIST_dir_0.01_train_100.pkl", "α=0.01 Train"),
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl", "α=0.01"),
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_iid.pkl", "IID")
]

fig, axes = plt.subplots(4, 2, figsize=(12, 20))

for i, (file_path, title_str) in enumerate(files):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # data is assumed to be a list where each element corresponds to a client's labels.
    # If they are nested or have irregular shapes, let's flatten them.
    all_labels_flat = []
    for labels in data:
        # Convert labels to a numpy array to ensure flattening
        arr = np.asarray(labels)
        
        # Flatten to 1D
        arr_flat = arr.ravel()

        # Extend the main list
        all_labels_flat.extend(arr_flat)

    # Now all_labels_flat should be a simple 1D list of labels.
    # Compute unique labels
    unique_labels = np.unique(all_labels_flat)
    num_classes = len(unique_labels)

    # Create a mapping from label to a consecutive integer
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    num_clients = len(data)
    client_matrix = np.zeros((num_clients, num_classes), dtype=int)

    # Fill the matrix
    for cli_idx, labels in enumerate(data):
        arr = np.asarray(labels).ravel()
        for lbl in arr:
            mapped_lbl = label_map[lbl]
            client_matrix[cli_idx, mapped_lbl] += 1

    # Compute proportions
    row_sums = client_matrix.sum(axis=1, keepdims=True)
    proportions = client_matrix / (row_sums + 1e-10)

    # Plot heatmap
    ax1 = axes[i, 0]
    im = ax1.imshow(proportions, aspect='auto', cmap='tab20')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Clients')
    ax1.set_title(f'MNIST Class Distribution ({title_str})')
    cbar = fig.colorbar(im, ax=ax1, label='Proportion')

    # Plot total samples per class
    ax2 = axes[i, 1]
    total_samples_per_class = client_matrix.sum(axis=0)
    ax2.bar(range(num_classes), total_samples_per_class, color='C0', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Total Samples')
    ax2.set_title('Total Samples per Class')

plt.tight_layout()
plt.savefig('MNIST_class_distributions_all.png', dpi=300)
plt.show()
"""
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

files = [
    (r"D:\STAT561\FL\dataset\MNIST\MNIST_dir_0.01_test_100.pkl", "alpha_0.01_test"),
    (r"D:\STAT561\FL\dataset\MNIST\MNIST_dir_0.01_train_100.pkl", "alpha_0.01_train"),
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl", "alpha_0.01"),
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_iid.pkl", "IID")
]

# Create output directory if not exists
os.makedirs('plots_separated', exist_ok=True)

for file_path, title_str in files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Convert each client's labels to a flat array
    all_labels_flat = []
    for labels in data:
        arr = np.asarray(labels).ravel()
        all_labels_flat.extend(arr)

    # Compute unique labels
    unique_labels = np.unique(all_labels_flat)
    num_classes = len(unique_labels)

    # Map labels to consecutive integers
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_clients = len(data)
    client_matrix = np.zeros((num_clients, num_classes), dtype=int)

    # Fill the matrix
    for cli_idx, labels in enumerate(data):
        arr = np.asarray(labels).ravel()
        for lbl in arr:
            mapped_lbl = label_map[lbl]
            client_matrix[cli_idx, mapped_lbl] += 1

    # Compute proportions
    row_sums = client_matrix.sum(axis=1, keepdims=True)
    proportions = client_matrix / (row_sums + 1e-10)

    # -------------------------
    # Heatmap Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(proportions, aspect='auto', cmap='tab20')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Clients')
    ax.set_title(f'MNIST Class Distribution ({title_str})')
    fig.colorbar(im, ax=ax, label='Proportion')
    
    # Adjust x-axis and y-axis
    # Classes: 0 to num_classes-1, ensure visible ticks
    ax.set_xticks(range(num_classes))
    # Clients: 0 to num_clients-1
    ax.set_yticks(range(0, num_clients, max(1, num_clients//10)))  # For large num_clients, step by some fraction
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_clients - 0.5, -0.5)  # To have client 0 at the top
    plt.tight_layout()
    
    heatmap_filename = f'plots_separated/MNIST_class_distribution_{title_str}_heatmap.png'
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()

    # -------------------------
    # Bar Plot for total samples per class
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    total_samples_per_class = client_matrix.sum(axis=0)
    ax.bar(range(num_classes), total_samples_per_class, color='C0', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Total Samples')
    ax.set_title(f'Total Samples per Class ({title_str})')
    
    # Adjust x-axis
    ax.set_xticks(range(num_classes))
    ax.set_xlim(-0.5, num_classes - 0.5)
    # For the y-axis, you can also do something like a log scale if needed:
    # ax.set_yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    bar_filename = f'plots_separated/MNIST_class_distribution_{title_str}_bar.png'
    plt.savefig(bar_filename, dpi=300)
    plt.close()

print("Plots have been saved in the 'plots_separated' directory.")

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Path to your data partition file, adjust as needed
file_path = r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 'data' is assumed to be a list of lists: data[i] gives the labels for client i.
# Convert all labels to a 2D array-like structure. 
# If clients have different lengths, we can just show them as rows of different length.

# We'll create a 2D array of shape (num_clients, max_samples_per_client),
# padding shorter clients with a special value if needed.
lengths = [len(d) for d in data]
max_length = max(lengths)
num_clients = len(data)

# Create an array filled with -1 (for padding)
arr = np.full((num_clients, max_length), -1, dtype=int)
for i, labels in enumerate(data):
    arr[i, :len(labels)] = labels

# Get all unique labels (except -1)
unique_labels = np.unique(arr[arr >= 0])
num_classes = len(unique_labels)

# Create a label map so labels are in the range [0, num_classes-1]
label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

# Map the array using label_map
mapped_arr = np.full_like(arr, -1)
for lbl, idx_map in label_map.items():
    mapped_arr[arr == lbl] = idx_map

# For padding (-1), we can assign a background color
# We'll use a colormap that has distinct colors. 'tab20' gives up to 20 distinct colors.
cmap = plt.get_cmap('tab20', num_classes + 1)  # one extra for padding
colors = cmap(np.arange(num_classes + 1))
# Make padding color white (or any light color)
colors[-1] = [1, 1, 1, 1]  # White
new_cmap = ListedColormap(colors)

# Values for imshow: we will map -1 to the last color, and classes to their indices
mapped_arr[mapped_arr == -1] = num_classes  # padding index

plt.figure(figsize=(10, 8))
plt.imshow(mapped_arr, aspect='auto', cmap=new_cmap)

# Turn off axes since user doesn't care about them
plt.xticks([])
plt.yticks([])
plt.box(False)

plt.title("Data Partition Visualization", fontsize=16)
plt.tight_layout()
plt.savefig("partition_visualization.png", dpi=300)
plt.show()
"""



"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Files to plot
files = [
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl", "Non-IID (α=0.01)"),
    (r"D:\STAT561\FL\dataset\data_partition_result\MNIST_iid.pkl", "IID")
]

def load_and_prepare_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Flatten each client's data and determine max length
    lengths = [len(np.ravel(labels)) for labels in data]
    max_length = max(lengths)
    num_clients = len(data)

    # Create a padded array for uniform shape
    arr = np.full((num_clients, max_length), -1, dtype=int)
    for i, labels in enumerate(data):
        flat_labels = np.ravel(labels)
        arr[i, :len(flat_labels)] = flat_labels

    # Map labels to a 0-based index
    unique_labels = np.unique(arr[arr >= 0])
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    mapped_arr = np.full_like(arr, -1)
    for lbl, idx_map in label_map.items():
        mapped_arr[arr == lbl] = idx_map

    # Map padding to the last index
    mapped_arr[mapped_arr == -1] = num_classes

    return mapped_arr, unique_labels, num_classes, label_map

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for ax, (file_path, title_str) in zip(axs, files):
    mapped_arr, unique_labels, num_classes, label_map = load_and_prepare_data(file_path)

    # Create a colormap for classes + 1 (for padding)
    cmap_base = plt.get_cmap('tab20', num_classes + 1)
    colors = cmap_base(np.arange(num_classes + 1))
    # Set padding color to white
    colors[-1] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)

    im = ax.imshow(mapped_arr, aspect='auto', cmap=cmap)

    # Axis labels: x-axis is sample index, y-axis is client index
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Client Index")
    ax.set_title(f"Data Partition Visualization ({title_str})")

# Create a single colorbar for both subplots
# We'll add the colorbar with discrete ticks corresponding to each class and one for padding
cbar = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.04)
# The last color is padding; we don't need a label for it.
# Set ticks for classes only
cbar.set_ticks(np.linspace(0.5, num_classes - 0.5, num_classes))
cbar.set_ticklabels([str(lbl) for lbl in unique_labels])
cbar.set_label("Class Label (Color represents different classes)")

plt.tight_layout()
plt.savefig("data_partition_comparison.png", dpi=300)
plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Example data simulation:
# Suppose we have 100 clients and 10 classes (0 through 9)
num_clients = 100
num_classes = 10

# Generate some fake data for demonstration:
# Each client gets random counts for each class
np.random.seed(0)
client_matrix = np.random.randint(0, 200, size=(num_clients, num_classes))

# Compute the total samples per class
total_samples_per_class = client_matrix.sum(axis=0)

# Normalize client_matrix to proportions if you want a color distribution
# (You can skip this if you prefer absolute counts)
row_sums = client_matrix.sum(axis=1, keepdims=True)
proportions = client_matrix / (row_sums + 1e-10)

# Create a colormap for the classes (0-9)
# We'll use a discrete colormap with exactly 10 distinct colors.
# 'tab10' provides 10 distinct colors, perfect for classes 0-9.
cmap = plt.get_cmap('tab10', num_classes)

# Create the figure with a grid of subplots
fig = plt.figure(figsize=(10, 8))

# Left subplot: clients (y-axis) vs classes (x-axis) proportions
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(proportions, aspect='auto', cmap=cmap)

ax1.set_xlabel("Classes")
ax1.set_ylabel("Clients")
ax1.set_title("Class Distribution by Client")

# Set ticks to show class labels on x-axis
ax1.set_xticks(np.arange(num_classes))
ax1.set_xticklabels(np.arange(num_classes))
# For the y-axis (clients), you can either show all or a subset
# Here we show every 10th client for readability
ax1.set_yticks(np.arange(0, num_clients, 10))
ax1.set_yticklabels(np.arange(0, num_clients, 10))

# Right subplot: bar chart of total samples per class
ax2 = fig.add_subplot(1, 2, 2)
bars = ax2.bar(np.arange(num_classes), total_samples_per_class, color=cmap.colors, edgecolor='black')
ax2.set_xlabel("Classes")
ax2.set_ylabel("Total Samples")
ax2.set_title("Total Samples per Class")
ax2.set_xticks(np.arange(num_classes))
ax2.set_xticklabels(np.arange(num_classes))

# Add a legend for classes below the figure
# We'll create a small legend block at the bottom
legend_labels = [str(cls) for cls in range(num_classes)]
legend_colors = [cmap(i) for i in range(num_classes)]

# Create a new axis at the bottom of the figure for the legend
legend_ax = fig.add_axes([0.25, 0.05, 0.5, 0.07])  # [left, bottom, width, height] in figure coordinates
legend_ax.set_axis_off()

# Plot a series of colored patches with labels
for i, (color, lbl) in enumerate(zip(legend_colors, legend_labels)):
    legend_ax.bar(i, 1, color=color, edgecolor='black')
legend_ax.set_xticks(np.arange(num_classes))
legend_ax.set_xticklabels(legend_labels)
legend_ax.set_yticks([])
legend_ax.set_title("Classes", pad=10)

# Adjust layout so the legend doesn't overlap with subplots
plt.subplots_adjust(bottom=0.2)

plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Path to the MNIST_dir_0.01.pkl file
file_path = r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 'data' should be a list of lists/arrays of labels:
# data[i] -> labels for client i
num_clients = len(data)

# Extract all unique labels
all_labels = []
for labels in data:
    all_labels.extend(labels)
unique_labels = np.unique(all_labels)
num_classes = len(unique_labels)

# Create a mapping from actual label to a consecutive class index (0 to num_classes-1)
label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

# Build the client_matrix: rows = clients, cols = classes
client_matrix = np.zeros((num_clients, num_classes), dtype=int)
for i, labels in enumerate(data):
    for lbl in labels:
        mapped_lbl = label_map[lbl]
        client_matrix[i, mapped_lbl] += 1

# Compute total samples per class
total_samples_per_class = client_matrix.sum(axis=0)

# Convert to proportions for the heatmap (optional)
row_sums = client_matrix.sum(axis=1, keepdims=True)
proportions = client_matrix / (row_sums + 1e-10)

# We'll use a colormap with distinct colors for each class
# If it's MNIST (0-9), we have 10 classes. 'tab10' is a good choice.
cmap = plt.get_cmap('tab10', num_classes)

fig = plt.figure(figsize=(10, 8))

# Left subplot: Class distribution by client (heatmap)
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(proportions, aspect='auto', cmap=cmap)
ax1.set_xlabel("Classes")
ax1.set_ylabel("Clients")
ax1.set_title("Class Distribution by Client")

# Set tick labels for classes
ax1.set_xticks(np.arange(num_classes))
ax1.set_xticklabels(unique_labels)  # Show actual label values if desired

# For y-axis, label a subset of clients for readability (e.g., every 10th)
if num_clients > 10:
    ax1.set_yticks(np.arange(0, num_clients, max(1, num_clients//10)))
    ax1.set_yticklabels(np.arange(0, num_clients, max(1, num_clients//10)))

# Right subplot: Total samples per class (bar chart)
ax2 = fig.add_subplot(1, 2, 2)
bars = ax2.bar(np.arange(num_classes), total_samples_per_class, color=[cmap(i) for i in range(num_classes)], edgecolor='black')
ax2.set_xlabel("Classes")
ax2.set_ylabel("Total Samples")
ax2.set_title("Total Samples per Class")
ax2.set_xticks(np.arange(num_classes))
ax2.set_xticklabels(unique_labels)

# Add a legend for the classes below the figure
legend_ax = fig.add_axes([0.25, 0.05, 0.5, 0.07])
legend_ax.set_axis_off()

for i in range(num_classes):
    legend_ax.bar(i, 1, color=cmap(i), edgecolor='black')
legend_ax.set_xticks(np.arange(num_classes))
legend_ax.set_xticklabels(unique_labels)
legend_ax.set_yticks([])
legend_ax.set_title("Classes", pad=10)

# Adjust layout so legend fits
plt.subplots_adjust(bottom=0.2)

plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Path to the MNIST_dir_0.01.pkl file
file_path = r"D:\STAT561\FL\dataset\data_partition_result\MNIST_dir_0.01.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

num_clients = len(data)

# Extract all labels
all_labels = []
for labels in data:
    all_labels.extend(labels)
unique_labels = np.unique(all_labels)

# Assume we only want 10 classes
# If there are more than 10 classes, truncate to first 10
# If exactly 10 classes are expected, this ensures only those are used
if len(unique_labels) > 10:
    unique_labels = unique_labels[:10]

num_classes = len(unique_labels)
label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

# Build the client_matrix
client_matrix = np.zeros((num_clients, num_classes), dtype=int)
for i, labels in enumerate(data):
    for lbl in labels:
        if lbl in label_map:  # Only count labels within our 10-class subset
            mapped_lbl = label_map[lbl]
            client_matrix[i, mapped_lbl] += 1

# Compute proportions
row_sums = client_matrix.sum(axis=1, keepdims=True)
proportions = client_matrix / (row_sums + 1e-10)

# Use 'tab10' for exactly 10 distinct colors
cmap = plt.get_cmap('tab10', num_classes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Plot the class distribution heatmap
im = ax1.imshow(proportions, aspect='auto', cmap=cmap)
ax1.set_xlabel("Classes")
ax1.set_ylabel("Clients")
ax1.set_title("Class Distribution by Client")

# Set class labels on x-axis
ax1.set_xticks(np.arange(num_classes))
ax1.set_xticklabels(unique_labels)

# For y-axis, show clients at intervals if needed
if num_clients > 10:
    ax1.set_yticks(np.arange(0, num_clients, max(1, num_clients//10)))
    ax1.set_yticklabels(np.arange(0, num_clients, max(1, num_clients//10)))

# Plot total samples per class (bar chart)
total_samples_per_class = client_matrix.sum(axis=0)
bars = ax2.bar(np.arange(num_classes), total_samples_per_class,
               color=[cmap(i) for i in range(num_classes)], edgecolor='black')
ax2.set_xlabel("Classes")
ax2.set_ylabel("Total Samples")
ax2.set_title("Total Samples per Class")
ax2.set_xticks(np.arange(num_classes))
ax2.set_xticklabels(unique_labels)

# Create a discrete colorbar for the classes
# We'll create boundaries between classes at each integer step
boundaries = np.arange(num_classes+1) - 0.5
norm = BoundaryNorm(boundaries, num_classes)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax1, fraction=0.05, pad=0.04, ticks=np.arange(num_classes))
cbar.ax.set_yticklabels(unique_labels)  # Label each colorbar tick with the class
cbar.set_label("Class Label")

plt.tight_layout()
plt.show()
