import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from smo import SVM, MultiSVM

# --------------------------------------------------------------------------
#  Task 3: Training MultiSVM with polynomial kernel for Iris dataset
# --------------------------------------------------------------------------

data = load_iris()
X_iris = data.data
y_iris = data.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)

multi_svm_poly = MultiSVM(kernel='poly')
multi_svm_poly.fit(X_train_iris, y_train_iris)
y = []
for x in X_test_iris:
    multi_svm_poly_predictions = multi_svm_poly.predict(x)
    y.append(multi_svm_poly_predictions)
multi_svm_poly_accuracy = accuracy_score(y_test_iris, y)

# For comparing with sklearn's SVC
svc_poly_iris = SVC(kernel='poly', C=1.0, degree=3, coef0=1, gamma='scale')
svc_poly_iris.fit(X_train_iris, y_train_iris)
svc_poly_iris_predictions = svc_poly_iris.predict(X_test_iris)
svc_poly_iris_accuracy = accuracy_score(y_test_iris, svc_poly_iris_predictions)

print("\nPolynomial Kernel (MultiSVM) Accuracy:", round(multi_svm_poly_accuracy * 100, 2), "%")
print("Polynomial Kernel (Sklearn's SVC) Accuracy:", round(svc_poly_iris_accuracy * 100, 2), "%")
