# Purpose: to train the LDA model on the petal length and width
# features and visualize the decision boundary.

import numpy as np
import matplotlib.pyplot as plt
from LinearDiscriminantAnalysis import LDAClassifier
from mlxtend.plotting import plot_decision_regions

# Load the data
data = np.load('data_iris_classifiers2.npz')
X_train = data['X_train_petal_std']
y_train = data['y_train']

y_train = y_train.astype(int)

# Initialize the LDA classifier
lda = LDAClassifier()

# Train the model
lda.fit(X_train, y_train)

# Save the model
lda.save("lda_weights2.npz")

# Plotting the decision boundary for training data
plot_decision_regions(X_train, y_train.ravel(), clf=lda, legend=2)
plt.title("LDA Decision Boundary")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.savefig("train_lda2.png")
plt.show()