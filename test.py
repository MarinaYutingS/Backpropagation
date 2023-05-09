import numpy as np
import matplotlib.pyplot as plt

# Define SoftPlus activation function
def softplus(x):
    return np.log(1 + np.exp(x))

# Define derivative of SoftPlus activation function
def d_softplus(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases
w1 = np.random.randn(2, 1)
b1 = np.random.randn(2, 1)
w2 = np.random.randn(1, 2)
b2 = np.random.randn(1, 1)

# Define training data
X_train = np.array([[0], [0.5], [1]])
Y_train = np.array([[0], [1], [0]])

# Define testing data
X_test = np.array([[0.1], [0.3], [0.7], [0.9]])
Y_test = np.array([[0], [1], [0], [0]])

# Define hyperparameters
learning_rate = 0.1
num_iterations = 10000

# Train the neural network
costs = []
for i in range(num_iterations):
    # Forward propagation
    Z1 = np.dot(w1, X_train.T) + b1
    A1 = softplus(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = Z2
    
    # Compute cost
    cost = np.mean((Y_train.T - A2)**2)
    costs.append(cost)
    
    # Backward propagation
    dZ2 = A2 - Y_train.T
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * d_softplus(Z1)
    dW1 = np.dot(dZ1, X_train)
    dB1 = np.sum(dZ1, axis=1, keepdims=True)
    
    # Update weights and biases
    w2 = w2 - learning_rate * dW2
    b2 = b2 - learning_rate * dB2
    w1 = w1 - learning_rate * dW1
    b1 = b1 - learning_rate * dB1

# Test the neural network
Z1_test = np.dot(w1, X_test.T) + b1
A1_test = softplus(Z1_test)
Z2_test = np.dot(w2, A1_test) + b2
A2_test = Z2_test

# Plot the results
plt.plot(X_train, Y_train, 'ro', label='Training Data')
plt.plot(X_test, Y_test, 'bo', label='Testing Data')
plt.plot(X_test, A2_test.T, 'g', label='Predicted Output')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Neural Network with 1 Hidden Layer (SoftPlus Activation Function)')
plt.show()
