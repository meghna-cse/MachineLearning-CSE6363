import numpy as np
from LogisticRegression import LogisticRegression

# Load the data
data = np.load('data_iris_classifiers1.npz')
X_train = data['X_train_allfeatures_std']
y_train = data['y_train']

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model
model.save("logistic_weights1.npz")