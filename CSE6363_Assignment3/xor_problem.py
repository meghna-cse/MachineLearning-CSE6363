import numpy as np
from neural_networks import *
import matplotlib.pyplot as plt

# 1. XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

def cross_entropy_loss(y_true, y_pred):
    # Small value to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_gradient(y_true, y_pred):
    # Gradient of the Cross-Entropy loss with respect to predictions
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

def threshold_predictions(predictions):
    return np.where(predictions >= 0.5, 1, 0)

# 2. Train the model
# This function will be responsible for training the model using either Sigmoid or Tanh activations.
def train_xor_model(activation_func, epochs=10000, learning_rate=0.1):
    model = Sequential()
    model.add(Linear(2, 2))
    
    if activation_func == "sigmoid":
        model.add(Sigmoid())
    elif activation_func == "tanh":
        model.add(Tanh())
    else:
        raise ValueError("Invalid activation function specified.")
    
    model.add(Linear(2, 1))
    model.add(Sigmoid())
    
    losses = []  # Store losses for plotting

    for epoch in range(epochs):        
        # Forward pass
        output = model.forward(X)
        
        # Compute loss
        loss = cross_entropy_loss(y, output)
        
        # Backward pass
        gradient = cross_entropy_gradient(y, output)
        model.backward(gradient)
        model.update(learning_rate)
        
        # Update weights and biases
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * layer.dweights
                layer.bias -= learning_rate * layer.dbias
        
        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        losses.append(loss)

    # Plotting the loss over epochs
    plt.plot(losses)
    plt.title(f'Training Loss with {activation_func} Activation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'xor_loss_with_{activation_func}_activation.png')
    plt.show()

    return model


# 3. Train and test the model using Sigmoid activation
print("Training with Sigmoid Activation...")
sigmoid_model = train_xor_model("sigmoid")
sigmoid_predictions = sigmoid_model.forward(X)
print("Predictions with Sigmoid Activation:", sigmoid_predictions)


# 4. Train and test the model using Tanh activation:
print("\nTraining with Tanh Activation...")
tanh_model = train_xor_model("tanh")
tanh_predictions = tanh_model.forward(X)
print("Predictions with Tanh Activation:", tanh_predictions)

# 5. Threshold predictions
print("Thresholded Predictions with Sigmoid Activation:")
print(threshold_predictions(sigmoid_predictions))

print("\nThresholded Predictions with Tanh Activation:")
print(threshold_predictions(tanh_predictions))

# 6. Save the trained model weights of the one with best accuracy
sigmoid_final_loss = cross_entropy_loss(y, sigmoid_predictions)
tanh_final_loss = cross_entropy_loss(y, tanh_predictions)

# Save weights of the best model
if sigmoid_final_loss < tanh_final_loss:
    print('Sigmoid model performed better. Saving weights...')
    save_weights(sigmoid_model, "XOR_solved.w")
else:
    print('Tanh model performed better. Saving weights...')
    save_weights(tanh_model, "XOR_solved.w")