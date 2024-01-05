from decision_tree import DecisionTree
import numpy as np

# AdaBoost classifier that combines weak learners to form a strong learner
class AdaBoost:

    # Initialization of AdaBoost with various parameters
    def __init__(self, weak_learner, num_learners, learning_rate):
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate


    # Fit the AdaBoost classifier from the training dataset
    def fit(self, X, y):
        # Initialize weights to uniform distribution as each sample starts with the same importance
        weights = np.full(len(X), 1/len(X))
        self.learners = []
        self.weights = []

        # Loop over the number of learners to be created and trained
        for i in range(self.num_learners):
            # Fit weak learner on weighted samples
            learner = self.weak_learner()
            learner.fit(X, y, sample_weight=weights)
            self.learners.append(learner)

            # Compute weighted error and update weights
            y_pred = learner.predict(X)
            error = np.sum(weights * (y_pred != y))
            alpha = self.learning_rate * np.log((1 - error) / error)
            weights = weights * np.exp(alpha * (y_pred != y))
            weights = weights / np.sum(weights)

            # Save weights for prediction
            self.weights.append(alpha)

            # Stop early if perfect fit achieved
            if error == 0:
                break

            
    def predict(self, X):
        predictions = np.ones(len(X))
        for i in range(len(self.learners)):
            learner = self.learners[i]
            alpha = self.weights[i]
            predictions += alpha * np.array(learner.predict(X))
        return np.sign(predictions)