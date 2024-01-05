import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting.decision_regions import plot_decision_regions

def linear_kernel(x1, x2):
        return x1.T @ x2

# -------------------------------------------------------------
#  SVM class
# -------------------------------------------------------------
class SVM():
    def __init__(self, kernel, c=1.0, tol=1e-3, maxiter=1000):
        self._k = None
        self._kernel = kernel
        self._tol = tol
        self._maxiter = maxiter
        
        if kernel == 'linear':
            self._k = linear_kernel
        elif kernel == 'poly':
            self._k = self.polynomial_kernel
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")


        self._c = c
        
    def _init_params(self):
        self._error_cache = np.zeros(self._data.shape[0])
        self._alphas = np.ones(self._data.shape[0]) * .1
        self._b = 0
        
        if self._kernel == 'linear':
            self._weights = np.random.rand(self._data.shape[1])

    #  Task2: Added as part of Non-linear SVM task
    def polynomial_kernel(self, x1, x2):
        x1 = np.array(x1).flatten()
        x2 = np.array(x2).flatten()
        return (np.dot(x1, x2) + 1) ** 3
    
    def predict_score(self, x):
        """Predicts a raw score (not classification)
        
        Arguments
            x, array (batch_size, n_features) - input samples.
        """
        u = 0
        if self._kernel == 'linear':
            u = self._weights @ x.T - self._b
        else:
            for i in range(self._data.shape[0]):
                u += self._targets[i] * self._alphas[i] * self._k(self._data[i], x)
            u -= self._b

        return u
        
    def predict(self, X):
        """Classifies input samples.
        
        Arguments
            x, array (batch_size, n_features) - input samples.
        """
        scores = np.array([self.predict_score(x) for x in X])
    
        if scores.ndim > 1:
            scores[scores < 0] = -1
            scores[scores >= 0] = 1
            return scores
        else:
            return np.where(scores < 0, -1, 1)

    def smo_step(self, i1, i2):
        if i1 == i2:
            return 0

        x1 = self._data[i1]
        x2 = self._data[i2]
        y1 = self._targets[i1]
        y2 = self._targets[i2]
        alpha1 = self._alphas[i1]
        alpha2 = self._alphas[i2]

        # Compute errors for x1 and x2
        e1 = self.predict_score(x1) - y1
        e2 = self.predict_score(x2) - y2

        s = y1 * y2

        if s == 1:
            L = max(0, alpha2 + alpha1 - self._c)
            H = min(self._c, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self._c, self._c + alpha2 - alpha1)

        if L == H:
            return 0

        k11 = self._k(x1, x1)
        k22 = self._k(x2, x2)
        k12 = self._k(x1, x2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = alpha2 + y2 * (e1 - e2) / eta

            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        
        #  Task1: the negative case
        elif eta < 0:
            f1 = y1 * (e1 + self._b) - alpha1 * k11 - alpha2 * k12
            f2 = y2 * (e2 + self._b) - s * alpha2 * k12 - alpha2 * k22
            
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
        
            Psi_L = L1 * f1 + L * f2 + 0.5 * L1**2 * k11 + 0.5 * L**2 * k22 + s * L * L1 * k12
            Psi_H = H1 * f1 + H * f2 + 0.5 * H1**2 * k11 + 0.5 * H**2 * k22 + s * H * H1 * k12
            
            eps = 1e-5
            if Psi_L < Psi_H - eps:
                a2_new = L
            elif Psi_L > Psi_H + eps:
                a2_new = H
            else:
                a2_new = alpha2

        else:
            a2_new = alpha2
            
        if np.abs(a2_new - alpha2) < 1e-3 * (a2_new + alpha2 + 1e-3):
            return 0

        alpha1_new = alpha1 + s * (alpha2 - a2_new)

        # Update threshold to reflect change in Lagrange multipliers
        b1 = e1 + y1 * (alpha1_new - alpha1) * k11 + y2 * (a2_new - alpha2) * k12 + self._b
        b2 = e2 + y1 * (alpha1_new - alpha1) * k12 + y2 * (a2_new - alpha2) * k22 + self._b
        self._b = (b1 + b2) / 2

        # Update weight vector to reflect change in a1 & a2, if SVM is linear
        if self._kernel == 'linear':
            self._weights = np.sum((self._targets * self._alphas)[:, None] * self._data, axis=0)
        
        # Store a1 and a2 in alpha array
        self._alphas[i1] = alpha1_new
        self._alphas[i2] = a2_new

        # update error cache using new multipliers
        for i in range(self._data.shape[0]):
            self._error_cache[i] = self.predict_score(self._data[i]) - self._targets[i]

        return 1

    def examine(self, i2):
        x2 = self._data[i2]
        y2 = self._targets[i2]
        alpha2 = self._alphas[i2]
        e2 = self.predict_score(x2) - y2
        r2 = e2 * y2

        # Heuristic for picking the first multiplier
        # -self._tol < r2 < self._tol when point i2 is on the boundary
        # See (12) in Platt's paper
        if (r2 < -self._tol and alpha2 < self._c) or (r2 > self._tol and alpha2 > 0):
            f_idxs = np.where((self._alphas != 0) & (self._alphas != self._c))[0]

            if len(f_idxs) > 1:
                # Hueristic for second multiplier: get i1 with largest absolute difference |e1 - e2|

                max_step = 0
                for i, v in enumerate(f_idxs):
                    if v == i2:
                        continue

                    if self._error_cache[v] == 0:
                        self._error_cache[v] = self.predict_score(self._data[v]) - self._targets[v]
                    step = np.abs(self._error_cache[v] - e2)

                    if step > max_step:
                        max_step = step
                        i1 = v

                if self.smo_step(i1, i2):
                    return 1
                
                # Loop over all non-zero and non-C alpha, starting at random point
                for i, v in enumerate(np.random.permutation(f_idxs)):
                    if self.smo_step(v, i2):
                        return 1
                
                # Loop over all possible i1, starting at a random point
                for i, v in enumerate(np.random.permutation(range(self._data.shape[0]))):
                    if v == i2:
                        continue
                    if self.smo_step(v, i2):
                        return 1
                
        return 0
    
    def fit(self, data, targets):
        self._data = data
        self._targets = targets
        
        self._init_params()
        
        n_changed = 0
        examine_all = True
        n_iter = 0
        
        while (n_changed > 0 or examine_all is True) and n_iter < self._maxiter:
            n_changed = 0
            n_iter += 1
            
            if examine_all is True:
                # loop over all training examples
                random_idxs = np.random.permutation(np.arange(data.shape[0]))
                for i in random_idxs:
                    n_changed += self.examine(i)
            else:
                # loop over examples where alpha is not 0 & not C
                f_idxs = np.where((self._alphas != 0) & (self._alphas != self._c))[0]
                random_idxs = np.random.permutation(f_idxs)
                for i, v in enumerate(random_idxs):
                    n_changed += self.examine(v)
            
            if examine_all is True:
                examine_all = False
            elif n_changed == 0:
                examine_all = True

# -------------------------------------------------------------
#  Task 3: MultiSVM class
# -------------------------------------------------------------
class MultiSVM:
    def __init__(self, kernel):
        self.kernel_type = kernel
        self.classifiers = {}  # Holds the binary classifiers

    def fit(self, data, labels):
        unique_classes = np.unique(labels)
        for cls in unique_classes:
            binary_labels = np.where(labels == cls, 1, -1)  
            binary_classifier = SVM(kernel=self.kernel_type)
            binary_classifier.fit(data, binary_labels)
            self.classifiers[cls] = binary_classifier 

    def predict_score(self, X):
        scores = {}
        for cls, classifier in self.classifiers.items():
            scores[cls] = classifier.predict_score(X)
        return scores

    def predict(self, X):
        # Check if X is a single sample
        single_sample = False
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)  # Reshape to make it a batch of size 1
            single_sample = True

        scores = self.predict_score(X)
        predictions = []

        for i in range(X.shape[0]):
            # If we're processing a single sample, avoid indexing into scalar scores
            if single_sample:
                sample_scores = scores
            else:
                sample_scores = {cls: scores[cls][i] for cls in scores}
            
            predictions.append(max(sample_scores, key=sample_scores.get))

        if single_sample:
            return predictions[0]
        else:
            return np.array(predictions)

