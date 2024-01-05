import numpy as np

# -------------------------------------------------------------
#  Task 1: Kernel Matrix that results in a kernel matrix that
#           is NOT positive semidefinite.
# -------------------------------------------------------------

# considering sigmoid kernel as an example for Task 1
def sigmoid_kernel(x1, x2, a=1, b=-1):
    """Sigmoid kernel function."""
    return np.tanh(a * np.dot(x1, x2) + b)

def is_positive_semi_definite(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)

x1 = np.array([1, 0])
x2 = np.array([0, 1])

K = np.array([
    [sigmoid_kernel(x1, x1), sigmoid_kernel(x1, x2)],
    [sigmoid_kernel(x2, x1), sigmoid_kernel(x2, x2)]
])

print("Kernel matrix:")
print(K)

if is_positive_semi_definite(K):
    print("The kernel matrix is positive semidefinite.")
else:
    print("The kernel matrix is NOT positive semidefinite.")
