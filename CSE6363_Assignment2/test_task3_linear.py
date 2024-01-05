import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from smo import MultiSVM

# --------------------------------------------------------------------------
#  Task 3: Training MultiSVM with linear kernel for Iris dataset
# --------------------------------------------------------------------------

data = load_iris()
X_iris = data.data
y_iris = data.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)

multi_svm_linear = MultiSVM(kernel='linear')
multi_svm_linear.fit(X_train_iris, y_train_iris)
multi_svm_linear_predictions = multi_svm_linear.predict(X_test_iris)
multi_svm_linear_accuracy = accuracy_score(y_test_iris, multi_svm_linear_predictions)

# For comparing with sklearn's SVC
svc_linear_iris = SVC(kernel='linear', C=1.0)
svc_linear_iris.fit(X_train_iris, y_train_iris)
svc_linear_iris_predictions = svc_linear_iris.predict(X_test_iris)
svc_linear_iris_accuracy = accuracy_score(y_test_iris, svc_linear_iris_predictions)

print("Linear Kernel (MultiSVM) Accuracy:", round(multi_svm_linear_accuracy * 100, 2), "%")
print("Linear Kernel (Sklearn's SVC) Accuracy:", round(svc_linear_iris_accuracy * 100, 2), "%")