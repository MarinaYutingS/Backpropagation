import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

setosa_list = []

def ReLu(x):
    return np.maximum(0,x)
    
class NeuralNetworkMultiInputs:
    def __init__(self):
        self.w1 = -2.5
        self.w2 = 0.6
        self.b1 = 1.6
        self.w3 = -0.1
        
        self.w4 = -1.5
        self.w5 = 0.4
        self.b2 = 0.7
        self.w6 = 1.5
        
        self.w7 = 2.4
        self.w8 = -5.2
        
        self.w9 = -2.2
        self.w10 = 3.7
        
        self.b3 = 0
        self.b4 = 0
        self.b5 = 1

    def forward(self, X1,X2):
        self.x1 = X1 * self.w1 + X2 * self.w2 + self.b1 
        self.y1 = ReLu(self.x1) 
        self.x2 = X1 * self.w4 + X2 * self.w5 + self.b2
        self.y2 = ReLu(self.x2) 
        
        self.y1_1 = self.y1 * self.w3
        self.y2_1 = self.y2 * self.w6
        self.y1_sum = self.y1_1 + self.y2_1
        self.y1_output = self.y1_sum + self.b3
        
        self.y1_2 = self.y1 * self.w7
        self.y2_2 = self.y2 * self.w8
        self.y2_sum = self.y1_2 + self.y2_2 
        self.y2_output = self.y2_sum + self.b4
        
        self.y1_3 = self.y1 * self.w9
        self.y2_3 = self.y2 * self. w10
        self.y3_sum = self.y1_3 + self.y2_3
        self.y3_output = self.y3_sum + self.b5
        
        return self.y3_output

# Initial the neural network
nn = NeuralNetworkMultiInputs()

# Define the training data
pedal_width = np.linspace(0,1,6)
sepal_width = np.linspace(0,1,6)

# Forward pass
for x2 in range(len(sepal_width)):
    for x1 in range(len(pedal_width)):
        setosa = nn.forward(pedal_width[x1],sepal_width[x2])
        setosa_list.append(setosa)
    

pedal_list = [elem for _ in range(6) for elem in pedal_width]
sepal_list = np.repeat(sepal_width,6).tolist()

# +++++++++++++++++++++++++++ PLOTTING +++++++++++++++++++++++++++#

# Reshape the data to create a grid
X, Y = np.meshgrid(np.unique(pedal_list), np.unique(sepal_list))
Z = np.reshape(setosa_list, X.shape)
print(Z)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set background color to transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Create a custom colormap with different colors

# colors = ['#006400', '#006400'] #green
# colors = ['#FF0000', '#FF0000'] #red
colors = ['#800080', '#800080'] #purple

cmap = plt.cm.colors.ListedColormap(colors)

# Create a mask for values equals to 0
mask = np.isclose(Z, 0)

# Set the values in Z based on the mask
Z_modified = np.where(mask, 0, Z)

# Plot the data points
ax.scatter(pedal_list, sepal_list, setosa_list,color='#800080')

# Plot the surface
ax.plot_trisurf(X.flatten(), Y.flatten(), Z_modified.flatten(),  cmap=cmap, alpha=0.5)

# Set labels and title
ax.set_xlabel('Pedal Width')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Virginica')
ax.set_title('Prediction Iris Kinds - Virginica')

# Show the plot
plt.show()
