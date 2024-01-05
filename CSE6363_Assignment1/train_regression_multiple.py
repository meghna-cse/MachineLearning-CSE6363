# Purpose: Trains your model to predict two features (petal length and width)
# simultaneously using two other features (sepal length and width). This will
# generate the model weight file and the loss plot for the multiple outputs model.

from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Load training data
X_train = np.load("X_train_multiple.npy")
y_train = np.load("y_train_multiple.npy")

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model weights
model.save("model_weights_multiple.npz")

# Plot training losses
plt.plot(model.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss for Multiple Outputs Model")
plt.savefig("train_loss_multiple.png")
plt.show()
