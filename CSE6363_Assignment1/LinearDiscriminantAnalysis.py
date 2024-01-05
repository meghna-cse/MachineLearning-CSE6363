"""Linear Discriminant Analysis Classifier.
    
This class implements the LDA algorithm to find the most
discriminative linear combination of features.
"""

import numpy as np

class LDAClassifier:
    def __init__(self):
        self.weights = None                     # Projection weights
        self.class_labels = None                # Store class labels
        self.class_means_transformed = None     # Store transformed class means

    # ------------------------------------------------------
    # Method Name: compute_means
    # Method Description: Calculate the class-wise means of
    # the data.
    # ------------------------------------------------------
    def compute_means(self, X, y):
        unique_labels = np.unique(y)
        calculated_means = {}
        for lbl in unique_labels:
            calculated_means[lbl] = np.mean(X[y == lbl], axis=0)
        return calculated_means


    # ------------------------------------------------------
    # Method Name: fit
    # Method Description: This function computes the weights
    # that maximize the between-class variance and minimize
    # the within-class variance.
    # ------------------------------------------------------
    def fit(self, X, y):
        n = X.shape[1]
        means = self.compute_means(X, y)

        # Compute within-class scatter matrix
        Sw = np.zeros((n, n))
        for label, mean_vec in means.items():
            class_sw = np.zeros((n, n))
            for row in X[y == label]:
                row = row.reshape(n, 1)
                mean_vec = mean_vec.reshape(n, 1)
                class_sw += (row - mean_vec).dot((row - mean_vec).T)
            Sw += class_sw

        # Compute between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        Sb = np.zeros((n, n))
        for label, mean_vec in means.items():
            n_label = X[y == label].shape[0]
            mean_vec = mean_vec.reshape(n, 1)
            overall_mean = overall_mean.reshape(n, 1)
            Sb += n_label * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        # Compute eigenvectors and eigenvalues
        matrix = np.linalg.pinv(Sw).dot(Sb)
        eigvals, eigvecs = np.linalg.eig(matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Select the eigenvector with the highest eigenvalue
        self.weights = eigvecs[:, 0].real
        self.weights /= np.linalg.norm(self.weights, ord=2)
        
        # Store class labels for future use in prediction
        self.class_labels = np.unique(y)
        
        # Calculate class means in the transformed space
        transformed = self.transform(X)
        self.class_means_transformed = {}
        for cls in self.class_labels:
            self.class_means_transformed[cls] = np.mean(transformed[y == cls])


    # ------------------------------------------------------
    # Method Name: transform
    # Method Description: Transform data using the learned
    # weights.
    # ------------------------------------------------------
    def transform(self, X):
        return np.dot(X, self.weights)


    # ------------------------------------------------------
    # Method Name: predict
    # Method Description: Predict class labels
    # ------------------------------------------------------
    def predict(self, X):    
        transformed = self.transform(X)
        predictions = []
        for val in transformed:
            closest_class = min(self.class_means_transformed.keys(), key=lambda cls: abs(self.class_means_transformed[cls] - val))
            predictions.append(closest_class)
        return np.array(predictions)
    
    
    # ------------------------------------------------------
    # Method Name: save
    # Method Description: Save the model parameters to a 
    # specified file.
    # ------------------------------------------------------
    def save(self, filepath):
        np.savez(filepath, weights=self.weights, class_means_transformed=self.class_means_transformed)


    # ------------------------------------------------------
    # Method Name: load
    # Method Description: Load model parameters from a file.
    # ------------------------------------------------------
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.weights = data['weights']
        self.class_means_transformed = data['class_means_transformed'].item()