import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# ==== define the activation function ==== #
def softPlus(x):
    return np.log(1 + np.exp(x))

# ==== define training data ==== #
input = [0, 0.5, 1]
observed = [0, 1, 0]

# ==== define the paramaters for the gradiant descent ==== #
max_iteration = 700
learning_rate = 0.1
difference_threshold = 1e-6
absolute_threshold = 0.002
step_size_b3 = []
step_size_w3 = []
step_size_w4 = []
step_size_b1 = []
step_size_b2 = []
step_size_w1 = []
step_size_w2 = []
b3_list=[]
w3_list=[]
w4_list=[]
w1_list=[]
w2_list=[]
b1_list=[]
b2_list=[]



# +++++++++++++++ Define functions for the Forward propagation +++++++++++++++ #

# ==== define the function to calculate an array of x-axis value ==== #
def calculate_x(input,w,b):
    x = [round((w * i + b),2) for i in input]
    return x

# ==== define the function to calculate an array of y-axis value ==== #
def calculate_y(x):
    return [softPlus(x_axis) for x_axis in x]

# ==== define the function with two nodes in one layer ==== #
def blackbox(input, w1,w2,w3,w4,b1,b2,b3):
    x1 = calculate_x(input,w1,b1)
    x2 = calculate_x(input,w2,b2)
    y1 = calculate_y(x1)
    y2 = calculate_y(x2)
    output = [round((y1[i] * w3 + y2[i] * w4 + b3),2) for i in range(len(input))]
    
    return output,y1,y2,x1,x2

# +++++++++++++++ Define functions for the Backword propagation +++++++++++++++ #

# ==== define the loss function ==== #
def loss(expected,predicted):
    d_array = [np.exp2(expected[i] - predicted[i]) for i in range(len(input))]
    return sum(d_array)

# ==== define the function to calculate step size and new value for the gradiant descent ==== #
def gradiant_descent_value(derivative, learning_rate, value):
    step_size = derivative * learning_rate
    value -= step_size
    value = round(value,3)
    return value,step_size

# ==== check if the change in step size is below the difference_threshold ==== #
def criteria(iteration,step_size, val_string):
    if iteration > 0 and iteration < max_iteration and np.abs(step_size[-1] - step_size[-2]) < difference_threshold:
        if np.abs(step_size[-1]) < absolute_threshold:
            print(val_string,"Converged after", iteration, "iterations")
            iteration = max_iteration + 10
    else:
        if val_string != 'b3':
            iteration += 1

    return iteration

# ==== calculate deritive (gradient) ==== #
def calc_derivative(predicted,y1,y2,w3,w4,x1,x2,input_1_0):
    if y1 is not None:
        derivative = [-2 * (observed[i] - predicted[i]) * y1[i] for i in range(len(input))]
    elif y2 is not None:
        derivative = [-2 * (observed[i] - predicted[i]) * y2[i] for i in range(len(input))]
    else: 
        derivative = [-2 * (observed[i] - predicted[i]) for i in range(len(input))]
    if w3 is not None:
        if input_1_0 == 1:
            derivative = [-2 * (observed[i] - predicted[i]) * w3 * np.exp(x1[i]) * np.reciprocal(1 + np.exp(x1[i])) * input[i] for i in range(len(input))]
        else:
            derivative = [-2 * (observed[i] - predicted[i]) * w3 * np.exp(x1[i]) * np.reciprocal(1 + np.exp(x1[i])) for i in range(len(input))]
    if w4 is not None:
        if input_1_0 == 1:
            derivative = [-2 * (observed[i] - predicted[i]) * w4 * np.exp(x2[i]) * np.reciprocal(1 + np.exp(x2[i])) * input[i] for i in range(len(input))]
        else:
            derivative = [-2 * (observed[i] - predicted[i]) * w4 * np.exp(x2[i]) * np.reciprocal(1 + np.exp(x2[i])) for i in range(len(input))]
    return derivative

# ==== calculate new value ==== #
def calc_new_value(derivative,old_value):
    gradiant = gradiant_descent_value(derivative, learning_rate,old_value)
    new_value = gradiant[0]
    step_size = gradiant[1]
    return new_value, step_size

# ==== define a function that makes sure b3 is converged first, then backwords, check the second last one if satisifies the criteria to converge ==== #

def calc_iteration(back,forward,step_size_forward,forward_string):
    if back > forward + 1:
        forward = criteria(forward,step_size_forward, forward_string)
    elif back != forward:
        forward += 1
    return forward

# ==== Execution: Run the training data in the black box ==== #

def update_weights_biases(w1,w2,w3,w4,b1,b2,b3):
    b3_iteration = 0
    w3_iteration = 0
    w4_iteration = 0
    b1_iteration = 0
    b2_iteration = 0
    w1_iteration = 0
    w2_iteration = 0
    while w1_iteration < max_iteration or w2_iteration < max_iteration:

        # ==== forward pass ====
        output = blackbox(input, w1,w2,w3,w4,b1,b2,b3)
        predicted = output[0]
        y1 = output[1]
        y2 = output[2]
        x1 = output[3]
        x2 = output[4]
        
        # ==== backword pass ====
        
        # calculate deritive
        d_ssr_b3 = sum(calc_derivative(predicted, None, None,None, None, None, None,0))
        d_ssr_w3 = sum(calc_derivative(predicted, y1, None,None, None, None, None,0))
        d_ssr_w4 = sum(calc_derivative(predicted, None, y2,None, None,None, None,0))
        d_ssr_w1 = sum(calc_derivative(predicted, None, None,w3, None,x1, None,1))
        d_ssr_w2 = sum(calc_derivative(predicted, None, None,None, w4,None,x2, 1))
        d_ssr_b1 = sum(calc_derivative(predicted, None, None,w3, None,x1, None,0))
        d_ssr_b2 = sum(calc_derivative(predicted, None, None,None, w4,None,x2, 0))

        # calculate the gradiant
        b3_gradiant = calc_new_value(d_ssr_b3,b3)
        w3_gradiant = calc_new_value(d_ssr_w3,w3)
        w4_gradiant = calc_new_value(d_ssr_w4,w4)
        
        w1_gradiant = calc_new_value(d_ssr_w1,w1)
        w2_gradiant = calc_new_value(d_ssr_w2,w2)

        b1_gradiant = calc_new_value(d_ssr_b1,b1)
        b2_gradiant = calc_new_value(d_ssr_b2,b2)
        
        # calculate step size
        step_size_b3.append(b3_gradiant[1])
        step_size_w3.append(w3_gradiant[1])
        step_size_w4.append(w4_gradiant[1])
        
        step_size_b1.append(b1_gradiant[1])
        step_size_b2.append(b2_gradiant[1])
        step_size_w1.append(w1_gradiant[1])
        step_size_w2.append(w2_gradiant[1])
        


        # Check if the step size (loss function) is close to 0 # The changes in step size are close to 0 as well.# b3 cannot check criteria of convergence until its every 100 steps
  
        if b3_iteration % 100 ==0:
            b3_iteration = criteria(b3_iteration,step_size_b3, 'b3')
            if b3_iteration != max_iteration + 10:
                b3_iteration += 1
        elif b3_iteration != max_iteration + 10:
            b3_iteration += 1

        w3_iteration = calc_iteration(b3_iteration,w3_iteration,step_size_w3,'w3')

        w4_iteration = calc_iteration(b3_iteration,w4_iteration,step_size_w4,'w4')
        b1_iteration = calc_iteration(w3_iteration,b1_iteration,step_size_b1,'b1')
        b2_iteration = calc_iteration(w4_iteration,b2_iteration,step_size_b2,'b2')
        w1_iteration = calc_iteration(b1_iteration,w1_iteration,step_size_w1,'w1')
        w2_iteration = calc_iteration(b2_iteration,w2_iteration,step_size_w2,'w2')
        
        # ==== update the weights and biases ====
        if b3_iteration < max_iteration:
            b3 = b3_gradiant[0]
        if w3_iteration < max_iteration:
            w3 = w3_gradiant[0]
        if w4_iteration < max_iteration:
            w4 = w4_gradiant[0]
        if w1_iteration < max_iteration:   
            w1 = w1_gradiant[0]
        if w2_iteration < max_iteration:
            w2 = w2_gradiant[0]
        if b1_iteration < max_iteration:
            b1 = b1_gradiant[0]
        if b2_iteration < max_iteration:
            b2 = b2_gradiant[0]
            
        b3_list.append(b3)
        w3_list.append(w3)
        w4_list.append(w4)
        w1_list.append(w1)
        w2_list.append(w2)
        b1_list.append(b1)
        b2_list.append(b2)
    print('b3=',b3,'iteration = ',b3_iteration)
    print('w3=',w3,'iteration = ',w3_iteration)
    print('w4=',w4,'iteration = ',w4_iteration)
    print('w1=',w1,'iteration = ',w1_iteration)
    print('w2=',w2,'iteration = ',w2_iteration)
    print('b1=',b1,'iteration = ',b1_iteration)
    print('b2=',b2,'iteration = ',b2_iteration)
        
    return b3_list,w3_list,w4_list,w1_list,w2_list,b1_list,b2_list


