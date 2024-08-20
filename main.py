import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU function
def relu(x):
    return np.maximum(0, x)


# Tanh function
def tanh(x):
    return np.tanh(x)


# Softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)  # Sum over the entire array


# Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


# Define the input values
x_values = np.array([-1, 0, 1])

sigmoid_values = sigmoid(x_values)
relu_values = relu(x_values)
tanh_values = tanh(x_values)
softmax_values = softmax(x_values)
leaky_relu_values = leaky_relu(x_values)

# Plotting the activation functions using line plots
plt.figure(figsize=(12, 6))

# Sigmoid
plt.subplot(2, 3, 1)
plt.plot(x_values, sigmoid_values, marker='o', color='blue')
plt.title("Sigmoid")
plt.grid()

# ReLU
plt.subplot(2, 3, 2)
plt.plot(x_values, relu_values, marker='o', color='red')
plt.title("ReLU")
plt.grid()

# Tanh
plt.subplot(2, 3, 3)
plt.plot(x_values, tanh_values, marker='o', color='green')
plt.title("Tanh")
plt.grid()

# Softmax
plt.subplot(2, 3, 4)
plt.plot(x_values, softmax_values, marker='o', color='purple')
plt.title("Softmax")
plt.grid()

# Leaky ReLU
plt.subplot(2, 3, 5)
plt.plot(x_values, leaky_relu_values, marker='o', color='orange')
plt.title("Leaky ReLU")
plt.grid()

plt.tight_layout()
plt.show()
