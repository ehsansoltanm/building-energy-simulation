import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the Pareto optimal solutions
file_path = "/home/ubuntu/pareto_optimal_solutions.csv"
df_pareto = pd.read_csv(file_path)

# Define the objectives (lower is better)
objectives = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]

# Normalize the objective values to a 0-1 scale (where 0 is best, 1 is worst)
scaler = MinMaxScaler()
df_pareto_normalized = pd.DataFrame(scaler.fit_transform(df_pareto[objectives]), columns=objectives)

# Calculate a simple score: average normalized objective value (lower is better)
# Alternative: Euclidean distance from the ideal point (0, 0, 0)
df_pareto["Normalized_Score"] = df_pareto_normalized.mean(axis=1)
# df_pareto["Distance_From_Ideal"] = np.sqrt((df_pareto_normalized**2).sum(axis=1))

# Select the solution with the minimum score (best balance)
best_solution = df_pareto.loc[df_pareto["Normalized_Score"].idxmin()]

print("Selected Best Solution (Balanced Approach):")
print(best_solution)

# Save the best solution details to a file
with open("/home/ubuntu/selected_solution.txt", "w") as f:
    f.write("Selected Best Solution (Balanced Approach):\n")
    f.write(best_solution.to_string())

print("\nBest solution details saved to /home/ubuntu/selected_solution.txt")
