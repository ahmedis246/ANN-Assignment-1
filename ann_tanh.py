import numpy as np
np.random.seed(42)

def tanh(x):
    return np.tanh(x)

i1 = 0.05
i2 = 0.10
inputs = np.array([i1, i2])

W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 2))

b1 = 0.5
b2 = 0.7

net_hidden = np.dot(W1, inputs) + b1
out_hidden = tanh(net_hidden)

net_output = np.dot(W2, out_hidden) + b2
final_output = tanh(net_output)

print("Hidden Layer Output:")
print(out_hidden)

print("\nFinal Output of the Network:")
print(final_output)