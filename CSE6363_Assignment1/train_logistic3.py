import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load the data
data = np.load('data_iris_classifiers3.npz')
X_train = data['X_train_sepal_std']
y_train = data['y_train']

y_train = y_train.astype(int)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model
model.save("logistic_weights3.npz")

# Plotting the decision boundary for training data
plot_decision_regions(X_train, y_train.ravel(), clf=model, legend=2)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.savefig("train_logistic3.png")
plt.show()