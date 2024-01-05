# Purpose: Prepares the dataset for a multi-output regression task.

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()

# Using sepal length and width as input
X = iris.data[:, :2] 

# Predicting petal length and width
y = iris.data[:, 2:]

# Splitting the dataset into 90% training and 10% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

np.save("X_train_multiple.npy", X_train)
np.save("X_test_multiple.npy", X_test)
np.save("y_train_multiple.npy", y_train)
np.save("y_test_multiple.npy", y_test)
