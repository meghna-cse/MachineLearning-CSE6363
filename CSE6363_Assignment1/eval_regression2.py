# Purpose: Evaluates the trained model on the test set and prints the mean squared error.

from LinearRegression import LinearRegression
import numpy as np
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:]  # Using only petal length and petal width
y = iris.data[:, 0].reshape(-1, 1)  # Predicting sepal length

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and load the model
model = LinearRegression()
model.load("model_weights2.npz")

# Evaluate the model
mse = model.score(X_test, y_test)
print(f"Mean Squared Error for Model 2: {mse}")
