import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

input_training = [0, 0.5, 1] #dosage as the input array
expected_training = [0, 1, 0]

input = np.linspace(0,1,1000)
# ++++++++++++++++++++++++++++++++++++++++++++++++++ Inside the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #
w1 = -34.4
w2 = -1.3
w3 = -2.52
w4 = 2.28
b1 = 2.14
b2 = 1.29
b3 = -0.58

# ==== define the activation function ==== #
def softPlus(x):
    return np.log(1 + np.exp(x))

# ==== define the function to calculate an array of x-axis value ==== #
def calculate_x(input,w,b):
    return [w * i + b for i in input]

# ==== define the function to calculate an array of y-axis value ==== #
def calculate_y(x,w):
    return [softPlus(x_axis) * w for x_axis in x]

x1 = calculate_x(input,w1,b1)
x2 = calculate_x(input,w3,b2)
y1 = calculate_y(x1,w2)
y2 = calculate_y(x2,w4)

# ++++++++++++++++++++++++++++++++++++++++++++++++++ Inside the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #

output = [y1[i] + y2[i] + b3 for i in range(len(input))]

# plot the curve only using input and output array

fig, ax = plt.subplots()

line, = ax.plot(input, output)
plt.scatter(input_training,expected_training)

plt.show()

