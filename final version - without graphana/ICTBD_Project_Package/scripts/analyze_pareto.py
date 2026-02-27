import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a boolean mask of the same shape as costs.shape[0],
                        False to return integer indices of pareto-efficient points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            # And keep self
            is_efficient[i] = True
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

# Load the simulation results
df = pd.read_csv("/home/ubuntu/upload/simulation_outputs.csv")

# Define the objectives to minimize
objectives = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]
costs = df[objectives].values

# Find the Pareto efficient points
pareto_mask = is_pareto_efficient(costs)
pareto_df = df[pareto_mask].copy()

# Save the Pareto optimal solutions
pareto_df.to_csv("/home/ubuntu/pareto_optimal_solutions.csv", index=False)

print(f"Found {len(pareto_df)} Pareto optimal solutions.")
print("Pareto optimal solutions saved to /home/ubuntu/pareto_optimal_solutions.csv")

# --- Visualization ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all points
ax.scatter(df[objectives[0]], df[objectives[1]], df[objectives[2]], c='blue', marker='o', label='Dominated Solutions')

# Plot Pareto optimal points
ax.scatter(pareto_df[objectives[0]], pareto_df[objectives[1]], pareto_df[objectives[2]], c='red', marker='x', s=100, label='Pareto Optimal Solutions')

ax.set_xlabel(objectives[0] + ' (J)')
ax.set_ylabel(objectives[1] + ' (J)')
ax.set_zlabel(objectives[2] + ' (J)')
ax.set_title('Pareto Front for Energy Optimization')
ax.legend()
plt.tight_layout()
plt.savefig("/home/ubuntu/pareto_front_3d.png")
print("Pareto front 3D plot saved to /home/ubuntu/pareto_front_3d.png")

plt.close(fig)

# Create pair plot for better 2D visualization
import seaborn as sns

pareto_df_melt = pareto_df.reset_index().melt(id_vars=['index'] + list(df.columns.difference(objectives)), value_vars=objectives, var_name='Objective', value_name='Value')

# Add a column to distinguish Pareto points for plotting
df['Pareto'] = 'Dominated'
df.loc[pareto_mask, 'Pareto'] = 'Optimal'

pair_plot = sns.pairplot(df, vars=objectives, hue='Pareto', markers=['o', 'X'], palette={'Dominated': 'blue', 'Optimal': 'red'}, plot_kws={'alpha': 0.6, 's': 50}, diag_kind='kde')
pair_plot.fig.suptitle('Pareto Front Pair Plot', y=1.02)
pair_plot.savefig("/home/ubuntu/pareto_front_pairplot.png")
print("Pareto front pair plot saved to /home/ubuntu/pareto_front_pairplot.png")

