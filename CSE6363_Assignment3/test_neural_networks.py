from neural_networks import *

if __name__ == "__main__":
    # Test the Linear layer
    print("...Testing Linear layer...")
    layer = Linear(3, 2)
    input_data = np.array([[1, 2, 3]])
    output = layer.forward(input_data)
    print("Output:", output)

    gradient = np.array([[2, 3]])
    input_gradient = layer.backward(gradient)
    print("Gradient w.r.t input:", input_gradient)

    # Test the Sigmoid activation function
    print("...Testing Sigmoid activation function...")
    sigmoid_layer = Sigmoid()
    sigmoid_output = sigmoid_layer.forward(output)
    print("Sigmoid Output:", sigmoid_output)

    sigmoid_gradient = sigmoid_layer.backward(np.array([[0.5, 0.5]]))
    print("Gradient w.r.t input after Sigmoid:", sigmoid_gradient)

    # Test the Tanh activation function
    print("...Testing Tanh activation function...")
    tanh_layer = Tanh()
    tanh_output = tanh_layer.forward(output)
    print("Tanh Output:", tanh_output)

    tanh_gradient = tanh_layer.backward(np.array([[0.5, 0.5]]))
    print("Gradient w.r.t input after Tanh:", tanh_gradient)

    # Test the Softmax function
    print("...Testing Softmax function...")
    softmax_layer = Softmax()
    softmax_output = softmax_layer.forward(output)
    print("Softmax Output:", softmax_output)

    softmax_gradient = softmax_layer.backward(np.array([[0.5, 0.5]]))
    print("Gradient w.r.t input after Softmax:", softmax_gradient)

    # Test the CrossEntropyLoss class
    print("...Testing CrossEntropyLoss class...")
    loss_layer = CrossEntropyLoss()
    dummy_target = np.array([[1, 0]])
    computed_loss = loss_layer.forward(softmax_output, dummy_target)
    print("Computed Loss:", computed_loss)

    loss_gradient = loss_layer.backward()
    print("Gradient w.r.t input after Loss:", loss_gradient)

    # Test the Sequential class
    print("...Testing Sequential class...")
    model = Sequential()
    model.add(Linear(2, 3))
    model.add(Tanh())
    model.add(Linear(3, 2))
    model.add(Softmax())

    input_data = np.array([[0.5, 0.2]])
    model_output = model.forward(input_data)
    print("Model Output:", model_output)

    gradient = np.array([[0.1, -0.1]])
    model_gradient = model.backward(gradient)
    print("Gradient w.r.t input after Model:", model_gradient)

    # Test the Sequential class
    print("...Testing Sequential class...")
    model = Sequential()
    model.add(Linear(2, 3))
    model.add(Tanh())
    model.add(Linear(3, 2))
    model.add(Softmax())

    input_data = np.array([[0.5, 0.2]])
    model_output = model.forward(input_data)
    print("Model Output:", model_output)

    gradient = np.array([[0.1, -0.1]])
    model_gradient = model.backward(gradient)
    print("Gradient w.r.t input after Model:", model_gradient)
