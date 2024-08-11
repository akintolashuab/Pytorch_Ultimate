from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data (2 features with different scales)
data = np.array([[1.0, 200.0],
                 [2.0, 300.0],
                 [3.0, 400.0],
                 [4.0, 500.0]])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform it
scaled_data = scaler.fit_transform(data)

print("Original data:\n", data)
print("Scaled data:\n", scaled_data)