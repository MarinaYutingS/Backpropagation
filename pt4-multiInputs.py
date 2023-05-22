import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

setosa_list = []
def ReLu(x):
    return np.maximum(0,x)
    
class NeuralNetworkMultiInputs:
    def __init__(self):
        self.w1 = -2.5
        self.w2 = 0.6
        self.b1 = 1.6

    def forward(self, X1,X2):
        self.x1 = X1 * self.w1 + X2 * self.w2 + self.b1 
        self.y1 = ReLu(self.x1) 
        return self.y1

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
    
# print(len(setosa_list))
pedal_list = [elem for _ in range(6) for elem in pedal_width]
sepal_list = np.repeat(sepal_width,6).tolist()

# +++++++++++++++++++++++++++ PLOTTING +++++++++++++++++++++++++++#

# Reshape the data to create a grid
X, Y = np.meshgrid(np.unique(pedal_list), np.unique(sepal_list))
Z = np.reshape(setosa_list, X.shape)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set background color to transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# Create a custom colormap with reversed colors
colors = ["darkblue", "lightblue"]
cmap = plt.cm.colors.ListedColormap(colors)
# Create a mask for values equals to 0
mask = Z == 0
# Set the values in Z based on the mask
Z_modified = np.where(mask, 1, 0)
# Plot the data points
ax.scatter(pedal_list, sepal_list, setosa_list)
# Plot the surface
ax.plot_trisurf(X.flatten(), Y.flatten(), Z_modified.flatten(),  cmap=cmap, alpha=0.5)

# Set labels and title
ax.set_xlabel('Pedal Width')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Setosa')
ax.set_title('3D Scatter Plot')

# Show the plot
plt.show()
