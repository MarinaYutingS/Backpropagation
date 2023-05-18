import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import mymodule

b3 = 0
w1 = 2.74
w2 = -1.13
w3 = 0.13
w4 = 0.63
b1 = 0
b2 = 0

b3_line = []
w3_line = []
w4_line = []
w1_line = []
w2_line = []
b1_line = []
b2_line = []

paramater_list= mymodule.update_weights_biases(w1,w2,w3,w4,b1,b2,b3)
b3_list = paramater_list[0]
w3_list = paramater_list[1]
w4_list = paramater_list[2]
w1_list = paramater_list[3]
w2_list = paramater_list[4]
b1_list = paramater_list[5]
b2_list = paramater_list[6]
# loop = (len(b3_list)-1)/50
# print('b3_list=',b3_list[0],b3_list[149],b3_list[299],b3_list[449],b3_list[599])
# put some paramaters into a list

for i in range(5):
    b3_line.append(b3_list[150*i])
    w3_line.append(w3_list[150*i])
    w4_line.append(w4_list[150*i])
    w1_line.append(w1_list[150*i])
    w2_line.append(w2_list[150*i])
    b1_line.append(b1_list[150*i])
    b2_line.append(b2_list[150*i])
    
# print('b3_line',b3_line)
# print('w3_line',w3_line)
# print('w4_line',w4_line)
# print('w1_line',w1_line)
# print('w2_line',w2_line)
# print('b1_line',b1_line)
# print('b2_line',b2_line)

# Generate x values
x = np.linspace(0, 1, 100)

# Define the initial function y = x^2
def func_init(x):
    return x ** 2

# Define the target function y = x^3
def func_target(x, i):
    # return x ** 3 + b3_line[i]
    # target = mymodule.blackbox(x,w1_line[i],w2_line[i],w3_line[i],w4_line[i],b1_line[i],b2_line[i],b3_line[i])[0]
    x1 = mymodule.calculate_x(x,w1_line[i],b1_line[i])
    x2 = mymodule.calculate_x(x,w2_line[i],b2_line[i])
    y1 = mymodule.softPlus(x1)
    y2 = mymodule.softPlus(x2)
    y = y1 * w3_line[i] + y2 * w4_line[i] + b3_line[i]
    return y

# Initialize the figure and axes
fig, ax = plt.subplots()
line, = ax.plot(x, func_init(x), color='blue')
ax.set_ylim(0, 1.25)
ax.set_xlim(0,1.25)
training = [0, 0.5, 1]
observed = [0, 1, 0]
plt.scatter(training,observed,color='red')
# Animation update function
def update(frame):
    i = int(frame * (len(b3_line)-1))
    # Compute the interpolated function values
    y_interp = (1 - frame) * func_init(x) + frame * func_target(x,i)
    
    # Update the line data
    line.set_ydata(y_interp)
    
    # Return the updated line
    return line,

# Create the animation
animation = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), interval=50, blit=True, repeat=False)

# Show the plot
plt.show()
