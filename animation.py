import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

b3_list= [-0.112, 0.604, 1.51, 2.392, 2.448]

# Generate x values
x = np.linspace(-10, 10, 100)

# Define the initial function y = x^2
def func_init(x):
    return x ** 2

# Define the target function y = x^3
def func_target(x, i):
    return x ** 3 + b3_list[i]

# Initialize the figure and axes
fig, ax = plt.subplots()
line, = ax.plot(x, func_init(x), color='blue')

# Animation update function
def update(frame):
    i = int(frame * (len(b3_list)-1))
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
