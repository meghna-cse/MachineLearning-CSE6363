# This file contains all the classes for the neural network library, 
# including the layers, activation functions, and the Sequential class.

import numpy as np
import json

# 1. Base Layer Class
class Layer:
    def __init__(self):
        pass

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

# 2. Linear Layer
class Linear(Layer):
    def __init__(self, input_nodes, output_nodes):
        super().__init__()
        # Initialize weights and bias
        self.weights = np.random.randn(input_nodes, output_nodes) * np.sqrt(2. / input_nodes)
        self.bias = np.zeros((1, output_nodes))
        
    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, output_gradient):
        self.dweights = np.dot(self.input_data.T, output_gradient)
        self.dbias = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)

# 3. Activation Functions

# 3.1. Sigmoid Function
class Sigmoid(Layer):
    def forward(self, input_data):
        input_data = np.clip(input_data, -20, 20)
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)


# 3.2. Hyperbolic Tangent Function
class Tanh(Layer):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (1 - self.output ** 2)

# 3.3. Softmax Function
class Softmax(Layer):
    def forward(self, input_data):
        exps = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        return self.output * (output_gradient - (output_gradient * self.output).sum(axis=1, keepdims=True))

# 4. CrossEntropyLoss
class CrossEntropyLoss(Layer):
    def forward(self, input_data, target):
        self.input_data = input_data
        self.target = target
        return -np.sum(target * np.log(input_data + 1e-7)) / input_data.shape[0]

    def backward(self):
        return (self.input_data - self.target) / self.input_data.shape[0]

# 5. Sequential Class
class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient
    
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)


# 6. Saving and Loading Model Weights
def save_weights(model, filename):
    data = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            data[f'weights_{i}'] = layer.weights.tolist()
            data[f'bias_{i}'] = layer.bias.tolist()
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_weights(model, filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            layer.weights = np.array(data[f'weights_{i}'])
            layer.bias = np.array(data[f'bias_{i}'])
