import numpy as np
import matplotlib.pyplot as plt

input = [0, 0.5, 1]
observed = [0, 1, 0]
b3 = 0
step_size = 100.000
learning_rate = 0.1
input_testing = np.linspace(0,1,1000)
# ++++++++++++++++++++++++++++++++++++++++++++++++++ START the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #

# ==== define the activation function ==== #
def softPlus(x):
    return np.log(1 + np.exp(x))

# ==== define the function to calculate an array of x-axis value ==== #
def calculate_x(input,w,b):
    return [w * i + b for i in input]

# ==== define the function to calculate an array of y-axis value ==== #
def calculate_y(x,w):
    return [softPlus(x_axis) * w for x_axis in x]

# ==== define the function with two nodes in one layer ==== #
def blackbox(input, w1,w2,w3,w4,b1,b2,b3):
    x1 = calculate_x(input,w1,b1)
    x2 = calculate_x(input,w3,b2)
    y1 = calculate_y(x1,w2)
    y2 = calculate_y(x2,w4)
    
    return [round(y1[i] + y2[i] + b3,2) for i in range(len(input))]

# +++++++++++++++++++++++++++ START the gradiant descent to calculate the optimized b3 +++++++++++++++++++++++++++ #
def optimize_b3(input,b3):
    for i in range(20):
        
        predicted = blackbox(input, 3.34,-1.22,-3.53,-2.3,-1.43,0.57,b3)

        d_ssr_b3 = [-2 * (observed[i] - predicted[i]) for i in range(len(input))]
        step_size = sum(d_ssr_b3) * learning_rate
        b3 = b3 - step_size
        b3= round(b3,3)
    return b3
# ++++++++++++++++++++++++++++++++++++++++++++++++++ END the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #
b3 = optimize_b3(input,b3)
output_testing = blackbox(input_testing, 3.34,-1.22,-3.53,-2.3,-1.43,0.57,b3)

fig, ax = plt.subplots()

plt.scatter(input,observed)

line, = ax.plot(input_testing, output_testing)



plt.show()