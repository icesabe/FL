import matplotlib.pyplot as plt
import numpy as np

# Data
# Format: { (dataset): {"round": round_number,
#                       "methods": {"comp_grad": value, "dp_comp_grad": value, "FedSTS": value}} }
data = {
    "MNIST(0.01)": {
        "round": 99,
        "methods": {
            "comp_grad": 56.00,
            "dp_comp_grad": 54.67,
            "FedSTS": 55.00
        }
    },
    "MNIST(0.001)": {
        "round": 99,
        "methods": {
            "comp_grad": 61.90,
            "dp_comp_grad": 52.14,
            "FedSTS": 52.62
        }
    },
    "CIFAR100(0.001)": {
        "round": 199,
        "methods": {
            "comp_grad": 23.21,
            "dp_comp_grad": 20.48,
            "FedSTS": 18.00
        }
    }
}

# We'll create one subplot per scenario in a column layout
fig, axes = plt.subplots(len(data), 1, figsize=(6, 6))

if len(data) == 1:
    axes = [axes]

scenarios = list(data.keys())

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    scenario_data = data[scenario]
    methods = list(scenario_data['methods'].keys())
    values = list(scenario_data['methods'].values())

    # Sort by value if desired (optional)
    # indices = np.argsort(values)
    # methods = [methods[j] for j in indices]
    # values = [values[j] for j in indices]

    # Create a horizontal bar plot
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, values, color='black', edgecolor='black')

    # Add text labels for the values on the bars
    for j, v in enumerate(values):
        ax.text(v + 0.5, j, f"{v:.2f}%", va='center', fontsize=9)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()  # To have the first method on top
    ax.set_xlabel('Accuracy (%)', fontsize=9)
    ax.set_title(f"{scenario}, Round={scenario_data['round']}", fontsize=10)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "MNIST(0.01)": {
        "round": 99,
        "methods": {
            "comp_grad": 56.00,
            "dp_comp_grad": 54.67,
            "FedSTS": 55.00
        }
    },
    "MNIST(0.001)": {
        "round": 99,
        "methods": {
            "comp_grad": 61.90,
            "dp_comp_grad": 52.14,
            "FedSTS": 52.62
        }
    },
    "CIFAR100(0.001)": {
        "round": 199,
        "methods": {
            "comp_grad": 23.21,
            "dp_comp_grad": 20.48,
            "FedSTS": 18.00
        }
    }
}

fig, axes = plt.subplots(len(data), 1, figsize=(6, 6))

if len(data) == 1:
    axes = [axes]

scenarios = list(data.keys())

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    scenario_data = data[scenario]
    methods = list(scenario_data['methods'].keys())
    values = list(scenario_data['methods'].values())

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, values, color='black', edgecolor='black')

    # Add text labels for the values on the bars
    for j, v in enumerate(values):
        ax.text(v + 0.5, j, f"{v:.2f}%", va='center', fontsize=9)

    # Find indices and values for FedSTS and comp_grad
    if "FedSTS" in methods and "comp_grad" in methods:
        fedsts_idx = methods.index("FedSTS")
        fedsts_val = scenario_data['methods']['FedSTS']
        comp_val = scenario_data['methods']['comp_grad']

        # Draw a red line at the FedSTS value
        ax.axvline(x=fedsts_val, color='red', linestyle='--', linewidth=1)
        
        # Compute improvement percentage
        improvement = ((comp_val - fedsts_val) / fedsts_val) * 100
        improvement_text = f"comp_grad is {improvement:.1f}% better than FedSTS"
        
        # Place the improvement text near the top of the subplot
        ax.text(fedsts_val + 1, fedsts_idx - 0.2, improvement_text, color='red', fontsize=8)

    # Formatting axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()  # So that the first method appears at the top
    ax.set_xlabel('Accuracy (%)', fontsize=9)
    ax.set_title(f"{scenario}, Round={scenario_data['round']}", fontsize=10)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

def plot_dirichlet(alpha, ax, title, epsilon=1e-5):
    """
    Plots the Dirichlet distribution on a 3D simplex.

    Parameters:
    - alpha: float, concentration parameter for a symmetric Dirichlet distribution.
    - ax: matplotlib 3D axis to plot on.
    - title: str, title of the plot.
    - epsilon: float, small positive number to avoid zero components.
    """
    # Number of points along each axis
    num_points = 50

    # Generate a grid of points in the simplex, excluding the boundaries
    x = np.linspace(epsilon, 1 - epsilon, num_points)
    y = np.linspace(epsilon, 1 - epsilon, num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Define the concentration parameters for a symmetric Dirichlet
    alpha_vector = [alpha, alpha, alpha]

    # Compute the Dirichlet PDF for each valid point in the simplex
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            y_val = Y[i, j]
            z_val = 1 - x_val - y_val
            # Ensure all components are greater than epsilon and sum to 1 within tolerance
            if z_val > epsilon:
                Z[i, j] = dirichlet.pdf([x_val, y_val, z_val], alpha_vector)
            else:
                Z[i, j] = np.nan  # Outside the valid simplex region

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Set plot labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$X_1$', fontsize=12)
    ax.set_ylabel('$X_2$', fontsize=12)
    ax.set_zlabel('PDF', fontsize=12)

    # Customize the z axis
    ax.set_zlim(0, np.nanmax(Z))
    ax.view_init(elev=30, azim=225)  # Adjust the viewing angle for better visualization

    # Optional: Add a color bar for reference
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Define alpha parameters for symmetric Dirichlet distributions
alpha_values = [0.01, 0.001]
titles = [r'Dirichlet Distribution ($\alpha = 0.01$)',
          r'Dirichlet Distribution ($\alpha = 0.001$)']

# Create side-by-side 3D plots
fig = plt.figure(figsize=(16, 7))

for i, alpha in enumerate(alpha_values):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    plot_dirichlet(alpha, ax, titles[i])

plt.tight_layout()
plt.show()
