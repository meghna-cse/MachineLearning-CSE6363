from LogisticRegression import LogisticRegression
from LinearDiscriminantAnalysis import LDAClassifier
import numpy as np

# Load test data
data = np.load('data_iris_classifiers1.npz')
X_test = data['X_test_allfeatures_std']
y_test = data['y_test']

# Initialize and load the LDA model, predict and evaluate
lda = LDAClassifier()
lda.load("lda_weights1.npz")
y_pred_lda = lda.predict(X_test)
accuracy_lda = np.mean(y_pred_lda == y_test.ravel())

# Initialize and load the Logistic Regression model, predict and evaluate
lr = LogisticRegression()
lr.load("logistic_weights1.npz")
y_pred_lr = lr.predict(X_test)
accuracy_lr = np.mean(y_pred_lr == y_test.ravel())

# Print
print(f"LDA Accuracy: {accuracy_lda * 100:.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")