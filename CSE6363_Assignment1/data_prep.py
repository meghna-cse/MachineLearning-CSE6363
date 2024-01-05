# Purpose: Prepare the dataset by loading and splitting it.

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Randomly split the dataset into 90% training and 10% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Save the datasets for further use
np.savez('iris_train_data.npz', X_train=X_train, y_train=y_train)
np.savez('iris_test_data.npz', X_test=X_test, y_test=y_test)
