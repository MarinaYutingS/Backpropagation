import numpy as np

# Create a 6x6 array of zeros (white pixels)
pixel_values = np.zeros((6, 6))
filter = np.zeros((3,3))

# Define pooling filter size
pool_size = (2, 2)
b1 = -2
# Define the Relu activation function
def ReLu(x):
    return np.maximum(0,x)

# ++++++++++++++++++++++++++++++++++++ Input ++++++++++++++++++++++++++++++++++++ #
# Set the specified positions to 1 (black pixels)
positions = [[0, 2], [1, 1], [2, 0], [3, 0], [4, 1], [5, 2], [0, 3], [1, 4], [2, 5], [3, 5], [4, 4], [5, 3]]
filter_positions = [[0,2],[1,1],[2,0]]
for pos in positions:
    pixel_values[pos[0], pos[1]] = 1

for pos in filter_positions:
    filter[pos[0],pos[1]] = 1

# ++++++++++++++++++++++++++++++++++++ Convolution ++++++++++++++++++++++++++++++++++++#
# Get the shape of the pixel_values array
num_rows, num_cols = pixel_values.shape
# Get the size of the sliding window
window_size = filter.shape
# Initialize the result array
feature_map_result = np.zeros((num_rows - window_size[0]+1,num_cols - window_size[1]+1))

# Iterate through the rows and columns with the filter sliding window
for i in range(num_rows - window_size[0]+1):
    for j in range(num_cols - window_size[1]+1):
        #Extract the current window
        window = pixel_values[i:i+window_size[0],j:j+window_size[1]]\
        #Perform the operation on the window
        feature_map = np.sum(np.multiply(window,filter)) + b1
        feature_map = ReLu(feature_map)
        feature_map_result[i,j] = feature_map

# Apply max pooling using np.max and np.reshape
pooled_array = np.max(np.reshape(feature_map_result, (2, 2, 2, 2)), axis=(1, 3))
input_nn = np.reshape(pooled_array,(1,4))

# ++++++++++++++++++++++++++++++++++++ Neural Network ++++++++++++++++++++++++++++++++++++ #
class NeuralNetwork:
    def __init__(self):
        self.W1 = np.array([[-0.8],[-0.07],[0.2],[0.17]])
        self.b1 = np.array([0.97])
        self.W2 = np.array([[-1.33,1.33]])
        self.b2 = np.array([[1.45,-0.45]])
    
    # Define the forward pass before the SoftMax or ArgMax
    def forward(self, X):
        self.z1 = np.dot(X,self.W1) + self.b1
        # print(self.z1.shape)
        self.a1 = ReLu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    # Loss Function
    
    # Training process & backpropogation

# ++++++++++++++++++++++++++++++++++++ Testing ++++++++++++++++++++++++++++++++++++ #

# Initialize the neural network
nn = NeuralNetwork()
result = nn.forward(input_nn)
print(result)
