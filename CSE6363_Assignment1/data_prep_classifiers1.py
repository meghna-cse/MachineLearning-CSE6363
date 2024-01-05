import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = datasets.load_iris()

# For all features
X_allfeatures = data.data
y = data.target

# Split the dataset into 90% training and 10% testing for all features
X_train_allfeatures, X_test_allfeatures, y_train, y_test = train_test_split(X_allfeatures, y, test_size=0.1, stratify=y, random_state=42)

# Standardize features
scaler_allfeatures = StandardScaler()
X_train_allfeatures_std = scaler_allfeatures.fit_transform(X_train_allfeatures)
X_test_allfeatures_std = scaler_allfeatures.transform(X_test_allfeatures)

# Save the datasets to be used in other scripts
np.savez('data_iris_classifiers1.npz', X_train_allfeatures_std=X_train_allfeatures_std, y_train=y_train, X_test_allfeatures_std=X_test_allfeatures_std, y_test=y_test)