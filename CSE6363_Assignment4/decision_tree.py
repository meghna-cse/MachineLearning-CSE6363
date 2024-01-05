import numpy as np

# DecisionTree classifier for building a decision tree based on given criterion
class DecisionTree:
    
    # Initialization of the DecisionTree with various parameters
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2,criterion="gini"):    
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.criterion_function = None # This will hold the actual function corresponding to the chosen criterion


    # Fit the decision tree classifier from the training dataset
    def fit(self, X, y,sample_weight=None):
      self.classes = len(np.unique(y))  # Determine the number of unique classes
      self.tree = self.buildTree(X, y)  # Build the tree using the buildTree method


    # Predict class or regression value for X
    # For each element in X, the predict method calls the recursive _predict method on the trained tree
    def predict(self, X):
      y_pred = []
      for inputs in X:
          output = self._predict(inputs)  # List comprehension to predict each sample
          y_pred.append(output)
      return y_pred  # Return the list of predictions


    # Private method to predict a single input sample using the trained tree
    def _predict(self, inputs):
        # This method would typically traverse the tree starting from the root and
        # following the path according to the input features until it reaches a leaf
        # node. Then it returns the prediction based on the value at the leaf node.

        node = self.tree
        while 'predicted_class' not in node:
            if inputs[node['splitting_feature']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']


    # Method to build the decision tree from the training data
    def buildTree(self, X, y, depth=0):
      # This method would involve choosing the best feature to split on based on the criterion,
      # creating a node in the tree, and recursively calling buildTree on the subsets of the data
      # created by the split, until the stopping criteria are met (max_depth, min_samples_split, 
      # min_samples_leaf).
      
      num_samples_per_class = []
      for i in range(self.classes):
        count = np.sum(y == i)
        num_samples_per_class.append(count)
      predicted_class = np.argmax(num_samples_per_class)
      node = {'depth': depth, 'n_samples': len(y), 'predicted_class': predicted_class}

      # Split recursively until maximum depth is reached or no more splits are possible
      if depth < self.max_depth:
          index, threshold = self.bestSplit(X, y)
          if index is not None:
              left_index = X[:, index] < threshold
              X_left = X[left_index] 
              y_left = y[left_index]
              X_right = X[~left_index]
              y_right = y[~left_index]
              if len(y_left) > self.min_samples_leaf and len(y_right) > self.min_samples_leaf:
                  node['splitting_feature'] = index
                  node['threshold'] = threshold
                  node['left'] = self.buildTree(X_left, y_left, depth + 1)
                  node['right'] = self.buildTree(X_right, y_right, depth + 1)
      return node
    
    def bestSplit(self, X, y):
      samples = len(y)
      if self.criterion == "entropy":
        criterion_function = self.entropy
      if self.criterion == "gini":
        criterion_function = self.gini
      if self.criterion == "missclassification":
        criterion_function = self.missclassification
      best_criterion = criterion_function(y)
      best_index = None
      best_threshold = None
      
      for index_b in range(X.shape[1]):
          thresholds, classes = self.calculate_th(X,y,index_b)
          
          num_left = [0] * self.classes #initialize it to 0 
          num_right = [np.sum(y == c) for c in range(self.classes)]
          for i in range(1, samples):
              c = classes[i - 1]
              num_left[c] += 1
              num_right[c] -= 1
              if i < self.min_samples_leaf or samples - i < self.min_samples_leaf:
                  continue
              criterion_left = criterion_function(num_left)
              criterion_right = criterion_function(num_right)
              criterion = (i * criterion_left + (samples - i) * criterion_right) / samples
              #print("criterion1",criterion_left,criterion_right)
              if thresholds[i] == thresholds[i - 1]:
                  continue
              #updating the variables that store the current best split found so far
              if criterion < best_criterion:
                  best_criterion = criterion
                  best_index = index_b
                  best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
      return best_index, best_threshold
    

    # Calculate the entropy of a dataset
    def entropy(self, y):
        # This function would calculate the entropy of the target values y, which is used
        # as a criterion to split the data in the tree.

        frequency_count = np.bincount(y)
        normalized_frequency_count = frequency_count / len(y)
        entropy = -np.sum([i * np.log2(i) for i in normalized_frequency_count if i > 0])
        return entropy
    
    def gini_impurity(self,y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        impurity = 1 - np.sum(np.square(probs))
        return impurity
    
    def gini(self, y):
      samples = len(y)
      gini = 1.0 - sum((np.sum(y == i) / samples) ** 2 for i in range(self.classes))
      return gini

    def missclassification(self, y):
        samples = len(y)
        missclassification = 1.0 - np.max([np.sum(y == i) / samples for i in range(self.classes)])
        return missclassification

    def calculate_threshold_classes(self,index_b,X,y):
      thresholds = []
      classes = []
      for i in range(len(y)):
              min_index = np.argmin(X[:, index_b])
              thresholds.append(X[min_index, index_b])
              classes.append(y[min_index])
              X = np.delete(X, min_index, axis=0)
              y = np.delete(y, min_index)
      return thresholds,classes

    def calculate_th(self,X,y,idx):
      return zip(*sorted(zip(X[:, idx], y)))