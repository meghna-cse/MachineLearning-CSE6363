# Pupose: Evaluates the trained model on the test data to
# provide you with the mean squared error for the multiple
# outputs model.

from LinearRegression import LinearRegression
import numpy as np

# Load training and test data
X_test = np.load("X_test_multiple.npy")
y_test = np.load("y_test_multiple.npy")

# Initialize and load the model
model = LinearRegression()
model.load("model_weights_multiple.npz")

# Evaluate the model
mse = model.score(X_test, y_test)
print(f"Mean Squared Error for Multiple Outputs Model: {mse}")
