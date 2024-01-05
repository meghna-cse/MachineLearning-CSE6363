# Necessary imports
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from smo import SVM


# -------------------------------------------------------------
#  Task 1: Training the SVM on linearly separable dataset
# -------------------------------------------------------------

X, y = make_blobs(n_samples=100, centers=2, random_state=6)
X[1] = X[0]
y[1] = y[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM(kernel='linear')
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, predictions)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")