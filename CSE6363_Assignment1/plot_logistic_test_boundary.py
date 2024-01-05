# Purpose: load the trained model, plot the decision boundary
# for the test data, and visualize the model's performance on
# unseen data

import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load the data
X_test = np.load("X_test.npy")
y_test = np.load("y_test_labels.npy")

y_test = y_test.astype(int)

# Considering only petal length and width as features
X_test = X_test[:, 2:]
#X_test = X_test[:, :2]  # Only sepal length and width

# Load the trained model
model = LogisticRegression()
model.load("logistic_weights2.npz")

y_test = y_test.reshape(-1, 1)

# Plotting the decision boundary for test data
plot_decision_regions(X_test, y_test.ravel(), clf=model, legend=2)
plt.title("Logistic Regression Decision Boundary (Test Data)")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.savefig("test_logistic2.png")
plt.show()
