import numpy as np
import mymodule

# ==== define the activation function ==== #
def ReLu(x):
    return max(0,x)
# ==== define training data ==== #
input = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
# input = np.linspace(0, 1, 100)
# ==== inialize values ==== #

w1,w2,w3,w4,b1,b2,b3 = (1.7, 12.6, -40.8, 2.7,-0.85, 0, -16)

# ==== define the function to calculate an array of y-axis value ==== #
def calculate_y(x):
    return [round(ReLu(x_axis),2) for x_axis in x]

# ==== forward propagation ==== #
x1 = mymodule.calculate_x(input,w1,b1)
x2 = mymodule.calculate_x(input,w2,b2)
y1 = calculate_y(x1)
y2 = calculate_y(x2)
y = [round((y1[i] * w3 + y2[i] * w4 + b3),2) for i in range(len(input))]
output = [ReLu(y_output) for y_output in y]
    
# print('y1=',y1)
# print('y2=',y2)
# print('output=',output)
