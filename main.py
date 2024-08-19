import numpy as np
import matplotlib.pyplot as plt


# Sigmoid:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU
def relu(x):
    return np.maximum(0, x)


# Tanh
def tanh(x):
    return np.tanh(x)


# Softmax (usually applied to a vector of values)
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


# Generate input data
x = np.linspace(-5, 5, 400)
x_matrix = x.reshape(-1, 1)  # For softmax (which usually expects 2D input)

# Plotting the activation functions
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x), label="Sigmoid", color='blue')
plt.title("Sigmoid")
plt.grid()

# ReLU
plt.subplot(2, 3, 2)
plt.plot(x, relu(x), label="ReLU", color='red')
plt.title("ReLU")
plt.grid()

# Tanh
plt.subplot(2, 3, 3)
plt.plot(x, tanh(x), label="Tanh", color='green')
plt.title("Tanh")
plt.grid()

# Softmax (since it’s typically applied to multiple values)
plt.subplot(2, 3, 4)
plt.plot(x, softmax(x_matrix).flatten(), label="Softmax", color='purple')
plt.title("Softmax")
plt.grid()

# Leaky ReLU
plt.subplot(2, 3, 5)
plt.plot(x, leaky_relu(x), label="Leaky ReLU", color='orange')
plt.title("Leaky ReLU")
plt.grid()

# Adjust layout
plt.tight_layout()
plt.show()
