# Purpose: to train the LDA model on all
# features and visualize the decision boundary.

import numpy as np
from LinearDiscriminantAnalysis import LDAClassifier

# Load the data
data = np.load('data_iris_classifiers1.npz')
X_train = data['X_train_allfeatures_std']
y_train = data['y_train']

y_train = y_train.astype(int)

# Initialize the LDA classifier
lda = LDAClassifier()

# Train the model
lda.fit(X_train, y_train)

# Save the model
lda.save("lda_weights1.npz")
