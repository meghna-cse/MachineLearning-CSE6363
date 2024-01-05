import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from neural_networks import Linear, Sigmoid, Sequential

# Load data
X_train, y_train, X_val, y_val, X_test = preprocess_data()

def compute_loss(predictions, targets):
    targets_array = targets.to_numpy().reshape(-1, 1)
    return np.mean((predictions - targets_array) ** 2)


def compute_gradient(predictions, targets):
    targets_array = targets.to_numpy().reshape(-1, 1)
    return 2 * (predictions - targets_array) / len(targets_array)



def train_model(model, X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=100):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X_train)
        loss = compute_loss(predictions, y_train)
        train_losses.append(loss)

        # Backward pass
        gradient = compute_gradient(predictions, y_train)
        model.backward(gradient)
        
        # Update weights
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.weights -= learning_rate * layer.dweights
                layer.bias -= learning_rate * layer.dbias
        
        # Validation
        val_predictions = model.forward(X_val)
        val_loss = compute_loss(val_predictions, y_val)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
    print()
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, title):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(f'trip_duration_loss_{title}')
    plt.show()


def get_test_predictions(model, X_test):
    return model.forward(X_test)


# Configuration 1
model1 = Sequential()
model1.add(Linear(5, 10))
model1.add(Sigmoid())
model1.add(Linear(10, 10))
model1.add(Sigmoid())
model1.add(Linear(10, 1))

print("Training Configuration 1...")
train_losses1, val_losses1 = train_model(model1, X_train, y_train, X_val, y_val, learning_rate=0.001)
test_predictions1 = get_test_predictions(model1, X_test)
plot_losses(train_losses1, val_losses1, 'Configuration 1 Losses')
print()

# Configuration 2
model2 = Sequential()
model2.add(Linear(5, 15))
model2.add(Sigmoid())
model2.add(Linear(15, 10))
model2.add(Sigmoid())
model2.add(Linear(10, 5))
model2.add(Sigmoid())
model2.add(Linear(5, 1))

print("Training Configuration 2...")
train_losses2, val_losses2 = train_model(model2, X_train, y_train, X_val, y_val, learning_rate=0.005)
test_predictions2 = get_test_predictions(model2, X_test)
plot_losses(train_losses2, val_losses2, 'Configuration 2 Losses')
print()

# Configuration 3
model3 = Sequential()
model3.add(Linear(5, 20))
model3.add(Sigmoid())
model3.add(Linear(20, 10))
model3.add(Sigmoid())
model3.add(Linear(10, 1))

print("Training Configuration 3...")
train_losses3, val_losses3 = train_model(model3, X_train, y_train, X_val, y_val, learning_rate=0.01)
test_predictions3 = get_test_predictions(model3, X_test)
plot_losses(train_losses3, val_losses3, 'Configuration 3 Losses')


# For predicting accuracy

# For Configuration 1
rmse_train1 = np.sqrt(train_losses1[-1])
rmse_val1 = np.sqrt(val_losses1[-1])
print(f"Configuration 1 Training RMSE: {rmse_train1:.4f}")
print(f"Configuration 1 Validation RMSE: {rmse_val1:.4f}\n")

# For Configuration 2
rmse_train2 = np.sqrt(train_losses2[-1])
rmse_val2 = np.sqrt(val_losses2[-1])
print(f"Configuration 2 Training RMSE: {rmse_train2:.4f}")
print(f"Configuration 2 Validation RMSE: {rmse_val2:.4f}\n")

# For Configuration 3
rmse_train3 = np.sqrt(train_losses3[-1])
rmse_val3 = np.sqrt(val_losses3[-1])
print(f"Configuration 3 Training RMSE: {rmse_train3:.4f}")
print(f"Configuration 3 Validation RMSE: {rmse_val3:.4f}\n")