# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load or create the dataset (using NumPy)
# Example data: 2D NumPy array with three variables (features) and multiple observations
data = np.array([
    [65, 70, 75],
    [68, 73, 80],
    [70, 72, 78],
    [66, 71, 76],
    [68, 69, 74]
])

# Convert the data to a Pandas DataFrame for better labeling
df = pd.DataFrame(data, columns=["Feature1", "Feature2", "Feature3"])

# Step 2: Calculate the covariance matrix using Pandas
cov_matrix = df.cov()

# Step 3: Visualize the covariance matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Covariance Matrix Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()

# Step 4: Show the heatmap
plt.show()
