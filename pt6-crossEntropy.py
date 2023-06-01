import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        
        self.W1 = np.array([[-2.5,-1.5],[0.6,0.4]])
        self.b1 = np.array([1.6,0.7])
        self.W2 = np.array([[-0.1,2.4,-2.2],[1.5,-5.2,3.7]])
        self.b2 = np.array([0,0,1])
    
    # Define the Relu activation function
    def ReLu(self, x):
        return np.maximum(0,x)
    
    # Define the SoftMax function
    def SoftMax(self, x):
        # exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        exps = np.exp(x)
        return exps / np.sum(exps,keepdims=True)
    
    # Define the ArgMax function
    def ArgMax(self,x):
        max_values = np.max(x,axis=1, keepdims=True)
        result = np.where(x == max_values, 1, 0)
        return result
    
    # Define the loss function Cross Entrophy
    # def CrossEntroy(self,x):
    #     return -1*np.log(x)
    
    def cross_entrophy_loss(self,y_true,y_pred):
        epsilon = 1e-8
        num_samples = y_true.shape[0]
        loss = -np.sum(np.log(y_pred[range(num_samples), y_true] + epsilon))
        return loss
    
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
            
            #  Calculate the loss
            loss = self.cross_entrophy_loss(y,y_pred)
            # Backpropagation
            
        return
            
# Training data
X_seto = np.array([0.04,0.42])
X_Virg = np.array([1,0.54])
X_Versi = np.array([0.5,0.37])


# Initialize the neural network
nn = NeuralNetwork()
softMax_seto = nn.SoftMax(nn.forward(X_seto))
ce_seto = nn.CrossEntroy(softMax_seto[0])
softMax_Virg = nn.SoftMax(nn.forward(X_Virg))
ce_Virg = nn.CrossEntroy(softMax_Virg[2])
softMax_versi = nn.SoftMax(nn.forward(X_Versi ))
ce_versi = nn.CrossEntroy(softMax_versi[1])
print('ce_seto=',ce_seto,'ce_virg=',ce_Virg,'ce_versi=',ce_versi)
ce_sum = ce_seto+ce_versi+ce_Virg
print('ce_sum=',ce_sum)
