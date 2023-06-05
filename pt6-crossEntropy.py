import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        
        self.W1 = np.array([[-2.5,-1.5],[0.6,0.4]])
        self.b1 = np.array([1.6,0.7])
        self.W2 = np.array([[-0.1,2.4,-2.2],[1.5,-5.2,3.7]])
        self.b2 = np.array([-2.00,0.00,1.00])
    
    # Define the Relu activation function
    def ReLu(self, x):
        return np.maximum(0,x)
    
    # Define the SoftMax function
    def SoftMax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps,keepdims=True,axis=1)
    
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
    
    # Define the loss function Cross Entrophy
    def CrossEntroy(self,x):
        return -1*np.log(x)
    
    # Define the training process
    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            #  Forward propagation
            y_pred = self.SoftMax(self.forward(X))
            
            #  Calculate the loss
            loss_b3 = sum(self.CrossEntroy(y_pred[range(len(X)),y[0]]))
            
            # Backpropagation
            delta2 = y_pred
            delta2 = np.sum(delta2, axis=0)-1
            
            # Update Weights and Biases
            self.b2 -= learning_rate * delta2
            
# Training data
X = np.array([[0.04,0.42],[1,0.54],[0.5,0.37]])
y = np.array([[0,2,1],[0,2,1],[0,2,1]])

# Initialize the neural network
nn = NeuralNetwork()

# Train the neural network
learning_rate = 1
num_epochs = 15
loss_b3 = nn.train(X,y,learning_rate,num_epochs)

# Predict on the testing data
y_predicted=nn.SoftMax(nn.forward(X))

