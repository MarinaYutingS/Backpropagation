import numpy as np
import matplotlib.pyplot as plt

input = [0, 0.5, 1]
observed = [0, 1, 0]
input_testing = np.linspace(0,1,5)

# ==== define the paramaters for the gradiant descent ==== #
max_iteration = 1000
learning_rate = 0.1
threshold = 1e-6
step_size_b3 = []
step_size_w3 = []
step_size_w4 = []
b3_iteration = 0
w3_iteration = 0
w4_iteration = 0

# ==== define the initial values of those to be optimized ==== #
b3 = 0
# w3 = round(np.random.randn(),2)
# w4 = round(np.random.randn(),2)
w3 = 0.36
w4 = 0.63

# b3 = 7.326
# w3 = -2.195
# w4 = -6.8
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
    x2 = calculate_x(input,w2,b2)
    y1 = calculate_y(x1,w3)
    y2 = calculate_y(x2,w4)
    output = [round((y1[i] + y2[i] + b3),2) for i in range(len(input))]
    
    return x1,x2,y1,y2,output

# +++++++++++++++++++++++++++ START the gradiant descent to calculate the optimized b3, w3, w4 +++++++++++++++++++++++++++ #

# ==== define the function to calculate step size and new value for the gradiant descent ==== #
def gradiant_descent_value(derivative, learning_rate, value):
    step_size = sum(derivative) * learning_rate
    value -= step_size
    value = round(value,3)
    print('step_size=',step_size)
    return value,step_size

# ==== check if the change in step size is below the threshold ==== #
def criteria(iteration,step_size,max_iteration):
    if iteration > 0 and np.abs(step_size[-1] - step_size[-2]) < threshold and np.abs(step_size[-1]) < 0.02:
        if np.abs(step_size[-1]) < 0.002:
            print("Converged after", iteration, "iterations")
            iteration = max_iteration + 10
    return iteration

# ==== calculate the predicted value ==== #
def calc_predicted(input,w3,w4,b3):
    output = blackbox(input, 3.34,-3.53,w3,w4,-1.43,0.57,b3)
    predicted = output[4]
    y1 = softPlus(output[0])
    y2 = softPlus(output[1])
    return predicted, y1,y2

# ==== define the function to calculate values need to be optimized ==== #
def optimize_b3(predicted,b3, iteration):
    
    d_ssr_b3 = [-2 * (observed[i] - predicted[i]) for i in range(len(input))]
        
    b3_gradiant = gradiant_descent_value(d_ssr_b3,learning_rate,b3)
    b3 = b3_gradiant[0]
    step_size_b3.append(b3_gradiant[1])     
     
    iteration = criteria(iteration,step_size_b3,max_iteration)
        
    # iteration += 1
    return b3, iteration

def optimize_w3(predicted,y1,w3,iteration):
    d_ssr_w3 = [-2 *  (observed[i] - predicted[i]) * y1[i] for i in range(len(input))]
    
    w3_gradiant = gradiant_descent_value(d_ssr_w3,learning_rate,w3)
    w3 = w3_gradiant[0]
    step_size_w3.append(w3_gradiant[1])
    
    iteration = criteria(iteration,step_size_w3,max_iteration)
    
    # iteration += 1
    return w3, iteration

def optimize_w4(predicted,y2,w4,iteration):
    y2 = softPlus(output[1])
    
    d_ssr_w4 = [-2 *  (observed[i] - predicted[i]) * y2[i] for i in range(len(input))]
    
    
    w4_gradiant = gradiant_descent_value(d_ssr_w4,learning_rate,w4)
    w4 = w4_gradiant[0]
    step_size_w4.append(w4_gradiant[1])       
    
    iteration = criteria(iteration,step_size_w4,max_iteration)
    
    # iteration += 1
    return w4, iteration

# ==== Execution: Run the training data in the black box ==== #
while b3_iteration < max_iteration and w3_iteration < max_iteration and w4_iteration < max_iteration:
    output = calc_predicted(input,w3,w4,b3)
    predicted = output[0]
    y1 = output[1]
    y2 = output[2]
    
    b3_output = optimize_b3(predicted,b3,b3_iteration)
    b3 = b3_output[0]
    b3_iteration = b3_output[1]
    
    w3_output = optimize_w3(predicted,y1,w3,w3_iteration)
    w3 = w3_output[0]
    w3_iteration = w3_output[1]
    
    w4_output = optimize_w4(predicted,y2,w4,w4_iteration)
    w4 = w4_output[0]
    w4_iteration = w4_output[1]
    
    print('b3=',b3,'w3=',w3,'w4=',w4)
    print()
    
    b3_iteration += 1
    w3_iteration += 1
    w4_iteration += 1
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++ END the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #


output_testing = blackbox(input_testing, 3.34,-3.53,w3,w4,-1.43,0.57,b3)[4]

fig, ax = plt.subplots()

plt.scatter(input,observed)

line, = ax.plot(input_testing, output_testing)

plt.show()