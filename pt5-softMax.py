import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        
        self.W1 = np.array([[-2.5,1.5],[0.6,0.4]])
        self.b1 = np.array([1.6,0.7])
        self.W2 = np.array([[-0.1,2.4,-2.2],[1.5,-5.2,3.7]])
        self.b2 = np.array([0,0,1])
        
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        
        # Initialize weights and biases
        # self.W1 = np.random.randn(input_dim, hidden_dim)
        # self.b1 = np.zeros(hidden_dim)
        # self.W2 = np.random.randn(hidden_dim, output_dim)
        # self.b2 = np.zeros(output_dim)
    
    # Define the Relu activation function
    def ReLu(self, x):
        return np.maximum(0,x)
    
    # Define the SoftMax function
    def SoftMax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1,keepdims=True)
    
    # Define the ArgMax function
    def ArgMax(self,x):
        max_values = np.max(x,axis=1, keepdims=True)
        result = np.where(x == max_values, 1, 0)
        return result
    
    # Define the forward pass before the SoftMax or ArgMax
    def forward(self, X):
        self.z1 = np.dot(X,self.W1) + self.b1
        self.a1 = self.ReLu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    # Define the training process
    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            #  Forward propagation
            y_pred = self.forward(X)
            
            # Backpropagation
            
        return
            
# Training data
X_train = np.zeros((36,2))
X_train[:,0] = np.tile(np.arange(0,1.2,0.2),6)[:36]
X_train[:,1] = np.repeat(np.arange(0,0.2*6,0.2),6)[:36]


# Initialize the neural network
nn = NeuralNetwork()
output = nn.forward(X_train)
argMax_output = nn.ArgMax(output)
softMax_output = nn.SoftMax(output)
