"""Linear Regression using Gradient Descent.

This class implements the LinearRegression which has
methods to train a linear model, predict with it, 
score its predictions, and save/load model weights.
"""

import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None     #will be set in fit method
        self.bias = None        #will be set in fit method
        self.learning_rate = 0.01 # Rate at which the model adjusts based on the error.
        self.losses = []        # Initialize losses as an empty list


    # ------------------------------------------------------
    # Method Name: fit
    # Method Description: initializes weights and biases 
    # based on the shape of X (input features) and y (output/target values).
    # ------------------------------------------------------
    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # If y has a shape (n_samples, 1), it's a single output regression. 
        # If it has a shape (n_samples, n_outputs) where n_outputs > 1, 
        # it's a multivariate regression.
        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        # Weights will have a shape of (number_of_features, number_of_outputs).
        # Bias will have a shape of (1, number_of_outputs).
        self.weights = np.random.randn(n_features, n_outputs)
        self.bias = np.zeros((1, n_outputs))

        # Dividing the data into training (90%) and validation (10%) sets.
        # It helps in monitoring the model's performance on unseen data 
        # and implementing early stopping.
        train_size = int(0.9 * n_samples)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        y_train = y_train.reshape(-1, 1)  # y_train is a column vector

        # Initialize best loss to infinity for early stopping
        best_loss = float('inf')   
        wait = 0

        # Loop through the data in batches and update the weights and 
        # bias based on the gradient of the loss function
        for epoch in range(self.max_epochs):
            indices = np.arange(train_size)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, train_size, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size].reshape(-1, 1)  # y_batch is a column vector

                # Calculate model's predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                # Mean squared error without regularization
                #loss = np.mean((y_pred - y_batch) ** 2)

                # Calculate the loss with regularization
                loss = np.mean((y_pred - y_batch) ** 2) + self.regularization * np.sum(self.weights ** 2)
                
                # Compute gradients
                gradient_w = (2 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
                gradient_b = 2 * np.mean(y_pred - y_batch, axis=0)

                # Update weights and biases
                #self.weights -= self.learning_rate * gradient_w
                self.weights -= self.learning_rate * (gradient_w + 2 * self.regularization * self.weights)
                self.bias -= self.learning_rate * gradient_b

            # Validation loss for early stopping:
            # After each epoch, compute the loss on the validation set
            y_val_pred = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((y_val_pred - y_val) ** 2)

            # Append the loss for this epoch to self.losses
            self.losses.append(loss)
            
            # Early stopping logic:
            # Implement early stopping based on the validation loss to prevent overfitting
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.weights = best_weights
                    self.bias = best_bias
                    break
    

    # ------------------------------------------------------
    # Method Name: predict
    # Method Description: Return the predicted values based
    # on the input data and trained weights and bias.
    # ------------------------------------------------------
    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        ### computes the dot product of the input data and the weights, then adds the bias
        return np.dot(X, self.weights) + self.bias


    # ------------------------------------------------------
    # Method Name: score
    # Method Description: Computes the mean squared error
    # between the predicted and actual values.
    # ------------------------------------------------------
    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse
    

    # ------------------------------------------------------
    # Method Name: save
    # Method Description: Save the weights and bias to a 
    # specified file.
    # ------------------------------------------------------
    def save(self, filepath):
        np.savez(filepath, weights=self.weights, bias=self.bias)


    # ------------------------------------------------------
    # Method Name: load
    # Method Description: Load model weights and bias from 
    # a file.
    # ------------------------------------------------------
    def load(self, filepath):
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']