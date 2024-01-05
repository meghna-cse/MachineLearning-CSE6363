import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from smo import SVM
import matplotlib
matplotlib.use('TkAgg')


# -------------------------------------------------------------
#  Task 2: Training SVM with polynomial kernel and visualizing
#           the result for non-linear dataset
# -------------------------------------------------------------

X, y = make_circles(n_samples=100, factor=0.3, noise=0.05, random_state=42)
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_poly = SVM(kernel='poly')
svm_poly.fit(X_train, y_train)
svm_poly_predictions = svm_poly.predict(X_test)
svm_poly_accuracy = np.mean(svm_poly_predictions == y_test) * 100

# For comparing with sklearn's SVC
svc_poly = SVC(kernel='poly', degree=3, coef0=1, C=5)
svc_poly.fit(X_train, y_train)
svc_poly_predictions = svc_poly.predict(X_test)
svc_poly_accuracy = np.mean(svc_poly_predictions == y_test) * 100

print(f"Polynomial Kernel (SVM) Accuracy: {svm_poly_accuracy:.2f} %")
print(f"Polynomial Kernel (Sklearn SVC) Accuracy: {svc_poly_accuracy:.2f} %")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_decision_regions(X_train, y_train, clf=svm_poly, legend=2)
plt.title('Polynomial Kernel (SVM)')
plt.subplot(1, 2, 2)
plot_decision_regions(X_train, y_train, clf=svc_poly, legend=2)
plt.title('Polynomial Kernel (Sklearn SVC)')
plt.tight_layout()
plt.savefig('test_task2_polynomial.png')
plt.show()