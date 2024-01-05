from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    
    # Initialization of the RandomForest with various parameters
    def __init__(self, input_classifier, num_trees, min_features):
        self.input_classifier = input_classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.trees = []                     # List to store the individual trees of the forest
        self.index_features = None          # Placeholder for indices of the features to be used
        

    # Fit the random forest classifier from the training dataset
    def fit(self, X, y):
        # Loop over the number of trees to be created
        for i in range(self.num_trees):
            sample = len(X)                 # Determine the number of samples in the dataset
            # Sample with replacement
            index = np.random.choice(sample, sample, replace=True)
            X_sample = X[index]
            y_sample = y[index]

            # Randomly select a subset of features for each tree
            index_features = np.random.choice(X.shape[1], self.min_features, replace=False)
            X_sample = X_sample[:, index_features]
            tree = self.input_classifier()
            tree.fit(X_sample, y_sample)

            # After fitting all trees, the random forest is ready to make predictions
            self.trees.append(tree)


    # Predict the class for each sample in X using the forest of trees
    def predict(self, X):
        # Each tree in the forest makes a prediction, and the final
        # prediction is the one that gets the most 'votes'.

        predictions = []
        for inputs in X:
            trees_predictions = []
            for tree in self.trees:
                # Make a prediction for each tree
                index_features = np.random.choice(X.shape[1], self.min_features, replace=False)
                tree_prediction = tree._predict(inputs[self.index_features])
                trees_predictions.append(tree_prediction)

            # Select the majority vote from all the tree predictions
            majority_vote = Counter(trees_predictions).most_common(X.shape[0])[0][0]
            predictions.append(majority_vote)
        return predictions