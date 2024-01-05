import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from smo import SVM

# -------------------------------------------------------------
#  Task 2: Training SVM with linear kernel and visualizing the
#           result for non-linear dataset
# -------------------------------------------------------------


X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

svm_linear = SVM(kernel='linear')
svm_linear.fit(X_train, y_train)
svm_linear_predictions = svm_linear.predict(X_test)
svm_linear_accuracy = accuracy_score(y_test, svm_linear_predictions)

# For comparing with sklearn's SVC
svc_linear = SVC(kernel='linear', C=1.0)
svc_linear.fit(X_train, y_train)
svc_linear_predictions = svc_linear.predict(X_test)
svc_linear_accuracy = accuracy_score(y_test, svc_linear_predictions)

print("Linear Kernel (SVM) Accuracy:", round(svm_linear_accuracy * 100, 2), "%")
print("Linear Kernel (Sklearn SVC) Accuracy:", round(svc_linear_accuracy * 100, 2), "%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_regions(X_test, y_test, clf=svm_linear, legend=2)
plt.title('Linear Kernel (SVM)')
plt.subplot(1, 2, 2)
plot_decision_regions(X_test, y_test, clf=svc_linear, legend=2)
plt.title('Linear Kernel (Sklearn SVC)')
plt.savefig('test_task2_linear.png')
plt.show()