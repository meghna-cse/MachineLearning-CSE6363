import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = datasets.load_iris()

# For petal features
X_petal = data.data[:, 2:]  # Only petal length and width
y = data.target

# Split the dataset into 90% training and 10% testing for petal features
X_train_petal, X_test_petal, y_train, y_test = train_test_split(X_petal, y, test_size=0.1, stratify=y, random_state=42)

# Standardize features
scaler_petal = StandardScaler()
X_train_petal_std = scaler_petal.fit_transform(X_train_petal)
X_test_petal_std = scaler_petal.transform(X_test_petal)

# Save the datasets to be used in other scripts
np.savez('data_iris_classifiers2.npz', X_train_petal_std=X_train_petal_std, y_train=y_train, X_test_petal_std=X_test_petal_std, y_test=y_test)