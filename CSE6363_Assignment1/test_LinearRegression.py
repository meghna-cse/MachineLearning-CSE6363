from LinearRegression import LinearRegression
import numpy as np

# Generate dummy data
X = np.random.randn(100, 4)  # 100 samples, 4 features
y = np.random.randn(100, 1)  # 100 samples, 1 output

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Print the first 5 losses to ensure it's working
print(model.losses[:5])
