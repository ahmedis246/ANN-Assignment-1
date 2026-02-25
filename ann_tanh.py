import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

X = np.array([[0.05, 0.10]])

Y = np.array([[0.01, 0.99]])

np.random.seed(42)

input_neurons = 2
hidden_neurons = 2
output_neurons = 2

W1 = np.random.uniform(-0.5, 0.5, (input_neurons, hidden_neurons))
W2 = np.random.uniform(-0.5, 0.5, (hidden_neurons, output_neurons))

b1 = np.full((1, hidden_neurons), 0.5)
b2 = np.full((1, output_neurons), 0.7)

learning_rate = 0.1
epochs = 5000

for epoch in range(epochs):

    # Forward Pass
    Z1 = np.dot(X, W1) + b1
    A1 = tanh(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = tanh(Z2)

    error = Y - A2

    dZ2 = error * tanh_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)

    dZ1 = np.dot(dZ2, W2.T) * tanh_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)

    W2 += learning_rate * dW2
    W1 += learning_rate * dW1
    b2 += learning_rate * dZ2
    b1 += learning_rate * dZ1

Z1 = np.dot(X, W1) + b1
A1 = tanh(Z1)

Z2 = np.dot(A1, W2) + b2
final_output = tanh(Z2)

print("Final Network Output:")
print(final_output)
