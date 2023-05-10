import numpy as np
import matplotlib.pyplot as plt

# ==== define the activation function ==== #
def softPlus(x):
    return np.log(1 + np.exp(x))

# ==== initialize weights and biases ==== #
b3 = 0
# w3 = 0.36
# w4 = 0.63

w3 = round(np.random.randn(),2)
w4 = round(np.random.randn(),2)
print('initial: w3=',w3, 'w4=',w4)

# ==== define training data ==== #
input = [0, 0.5, 1]
observed = [0, 1, 0]

# ==== define testing data ==== #
input_testing = np.linspace(0,1,5)

# ==== define the paramaters for the gradiant descent ==== #
max_iteration = 200
learning_rate = 0.1
difference_threshold = 1e-6
absolute_threshold = 0.002
step_size_b3 = []
step_size_w3 = []
step_size_w4 = []
b3_iteration = 0
w3_iteration = 0
w4_iteration = 0

# +++++++++++++++ Forward propagation +++++++++++++++ #

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

# ==== calculate the predicted value ==== #
def calc_predicted(input,w3,w4,b3):
    output = blackbox(input, 3.34,-3.53,w3,w4,-1.43,0.57,b3)
    predicted = output[4]
    y1 = softPlus(output[0])
    y2 = softPlus(output[1])
    return predicted, y1,y2


# ==== define the function to calculate step size and new value for the gradiant descent ==== #
def gradiant_descent_value(derivative, learning_rate, value):
    step_size = derivative * learning_rate
    value -= step_size
    value = round(value,3)
    # print('step_size=',step_size)
    
    return value,step_size

# ==== check if the change in step size is below the difference_threshold ==== #
def criteria(iteration,step_size, val_string):
    # print('criteria step_size[-1]=', step_size[-1])
    if iteration > 0 and iteration < max_iteration and np.abs(step_size[-1] - step_size[-2]) < difference_threshold:
        if np.abs(step_size[-1]) < absolute_threshold:
            print(val_string,"Converged after", iteration, "iterations, previous step size= ",round(step_size[-2],3),"current step size= ",round(step_size[-1],3))
            iteration = max_iteration + 10
    return iteration

# ==== calculate deritive ==== #
def calc_derivative(predicted,y1,y2):
    if y1 is not None:
        derivative = [-2 * (observed[i] - predicted[i]) * y1[i] for i in range(len(input))]
    elif y2 is not None:
        derivative = [-2 * (observed[i] - predicted[i]) * y2[i] for i in range(len(input))]
    else: 
        derivative = [-2 * (observed[i] - predicted[i]) for i in range(len(input))]
    return derivative

# ==== calculate new value ==== #
def calc_new_value(derivative,old_value):
    gradiant = gradiant_descent_value(derivative, learning_rate,old_value)
    new_value = gradiant[0]
    step_size = gradiant[1]
    return new_value, step_size

# ==== Execution: Run the training data in the black box ==== #
while b3_iteration < max_iteration or w3_iteration < max_iteration or w4_iteration < max_iteration:
    output = calc_predicted(input,w3,w4,b3)
    predicted = output[0]
    y1 = output[1]
    y2 = output[2]
    
    # calculate deritive
    d_ssr_b3 = sum(calc_derivative(predicted, None, None))
    d_ssr_w3 = sum(calc_derivative(predicted, y1, None))
    d_ssr_w4 = sum(calc_derivative(predicted, None, y2))

    # calculate the gradiant
    b3_gradiant = calc_new_value(d_ssr_b3,b3)
    w3_gradiant = calc_new_value(d_ssr_w3,w3)
    w4_gradiant = calc_new_value(d_ssr_w4,w4)

    # calculate step size
    step_size_b3.append(b3_gradiant[1])
    step_size_w3.append(w3_gradiant[1])
    step_size_w4.append(w4_gradiant[1])

    # compare step size to the criteria to see if new values need to be calculated
    b3_iteration = criteria(b3_iteration,step_size_b3, 'b3')
    w3_iteration = criteria(w3_iteration,step_size_w3, 'w3')
    w4_iteration = criteria(w4_iteration,step_size_w4, 'w4')
    
    # calculate new value
    if b3_iteration < max_iteration:
        b3 = b3_gradiant[0]
    if w3_iteration < max_iteration:
        w3 = w3_gradiant[0]
    if w4_iteration < max_iteration:
        w4 = w4_gradiant[0]
    
    # loop if the criteria is not true
    if b3_iteration != max_iteration + 10:
        b3_iteration += 1
    if w3_iteration != max_iteration + 10:
        w3_iteration += 1
    if w4_iteration != max_iteration + 10:
        w4_iteration += 1

print('b3=',b3, 'iteration = ',b3_iteration)
print('w3=',w3,'iteration = ',w3_iteration)
print('w4=',w4,'iteration = ',w4_iteration)
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++ END the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #



output_testing = blackbox(input_testing, 3.34,-3.53,w3,w4,-1.43,0.57,b3)[4]

fig, ax = plt.subplots()

plt.scatter(input,observed)

line, = ax.plot(input_testing, output_testing)

plt.show()