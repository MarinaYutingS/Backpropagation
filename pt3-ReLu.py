import numpy as np
import mymodule
import matplotlib.pyplot as plt

# ==== define the activation function ==== # (Variables)
def ReLu(x):
    return np.maximum(0,x)
# def d_Relu(x):
#     return np.where(x > 0, 1, 0)
# ==== define the function to calculate an array of y-axis value ==== # (Redefined based on activation function)
def calculate_y(x):
    return [round(ReLu(x_axis),2) for x_axis in x]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ==== define training data ==== # (Variables)
# input = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]

# # ==== inialize values ==== # (Outputs)
# w1,w2,w3,w4,b1,b2,b3 = (1.7, 12.6, -40.8, 2.7,-0.85, 0, 0)

# # ==== forward propagation ==== #
# x1 = mymodule.calculate_x(input,w1,b1)
# x2 = mymodule.calculate_x(input,w2,b2)
# y1 = calculate_y(x1)
# y2 = calculate_y(x2)
# y = [round((y1[i] * w3 + y2[i] * w4 + b3),2) for i in range(len(input))]
# output = [ReLu(y_output) for y_output in y]

# print(output)

#==== Define the neural network architecture ==== # 
class NeuralNetwork:
    def __init__(self):
        self.w1 = 1.7
        self.w2 = 12.6
        self.w3 = -40.8
        self.w4 = 2.7
        self.b1 = -0.85
        self.b2 = 0
        self.b3 = -16
    def forward(self, X):
        self.x1 = mymodule.calculate_x(X,self.w1,self.b1)
        self.x2 = mymodule.calculate_x(X,self.w2,self.b2)
        self.y1 = calculate_y(self.x1)
        self.y2 = calculate_y(self.x2)
        self.y = [round((self.y1[i] * self.w3 + self.y2[i] * self.w4 + self.b3),2) for i in range(len(X))]
        self.output = [ReLu(y_output) for y_output in self.y]
        return self.output
    def backward(self, X, Y, learning_rate):
        #compute gradiants  
        db3 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) for i in range(len(X))]
        dw3 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.y1[i] for i in range(len(X))]
        dw4 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.y2[i] for i in range(len(X))]
        db1 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.w3 * np.where(self.x1[i] > 0, 1, 0) for i in range(len(X))]
        db2 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.w4 * np.where(self.x2[i] > 0, 1, 0) for i in range(len(X))]
        dw1 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.w3 * np.where(self.x1[i] > 0, 1, 0) * X[i] for i in range(len(X))]
        dw2 = [-2 * (Y[i] - self.y[i]) * np.where(self.y[i] > 0, 1, 0) * self.w4 * np.where(self.x2[i] > 0, 1, 0) * X[i] for i in range(len(X))]

        # update weights and biases
        self.b3 -= learning_rate * sum(db3)
        self.w3 -= learning_rate * sum(dw3)
        self.w4 -= learning_rate * sum(dw4)
        self.b1 -= learning_rate * sum(db1)
        self.b2 -= learning_rate * sum(db2)
        self.w1 -= learning_rate * sum(dw1)
        self.w2 -= learning_rate * sum(dw2)
        
# define the training data
X = [0, 0.5, 1]
y = [0, 1.01, 0]

# Initial the neural network
nn = NeuralNetwork()

#Training the nn with SGD
epoches = 1000
learning_rate = 0.1

for epoch in range(epoches):
    # Forward pass
    y_predicted = nn.forward(X)
    
    # Compute loss (Sum of squared residuals)
    loss = sum([(y[i] - y_predicted[i]) ** 2 for i in range(len(X))])
    
    # Backward pass
    nn.backward(X, y, learning_rate)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
print('w1=',nn.w1,'w2=',nn.w2,'w3=',nn.w3,'w4=',nn.w4,'b1=',nn.b1,'b2=',nn.b2,'b3=',nn.b3)
        
# Test the trained neural network
x = np.linspace(0, 1, 100)
predictions = nn.forward(x)

# ================ Plot ===================
fig, ax = plt.subplots()
plt.scatter(X,y,color='red')
line, = ax.plot(x, predictions, color='green')
plt.show()
