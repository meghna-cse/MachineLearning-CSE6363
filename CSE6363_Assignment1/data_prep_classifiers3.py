import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = datasets.load_iris()

# For sepal features
X_sepal = data.data[:, :2]  # Only sepal length and width
y = data.target

# Split the dataset into 90% training and 10% testing for sepal features
X_train_sepal, X_test_sepal, y_train, y_test = train_test_split(X_sepal, y, test_size=0.1, stratify=y, random_state=42)

# Standardize features
scaler_sepal = StandardScaler()
X_train_sepal_std = scaler_sepal.fit_transform(X_train_sepal)
X_test_sepal_std = scaler_sepal.transform(X_test_sepal)

# Save the datasets to be used in other scripts
np.savez('data_iris_classifiers3.npz', X_train_sepal_std=X_train_sepal_std, y_train=y_train, X_test_sepal_std=X_test_sepal_std, y_test=y_test)