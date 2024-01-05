# Purpose: Loads the Iris dataset, splits it, trains the model on
# a combination of features, saves the model weights, and plots 
# the training loss.

from sklearn import datasets
from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only sepal length and sepal width
y = iris.data[:, 2].reshape(-1, 1)  # Predicting petal length

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model weights
model.save("model_weights1.npz")

# Print the coefficients and bias
coefficients = model.weights
bias = model.bias
print("Coefficients:", coefficients)
print("Bias:", bias)

# Plot training losses
plt.plot(model.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss for Model 1")
plt.savefig("train_loss1.png")
plt.show()
