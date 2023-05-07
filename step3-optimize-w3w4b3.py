import numpy as np
import matplotlib.pyplot as plt

input = [0, 0.5, 1]
observed = [0, 1, 0]
input_testing = np.linspace(0,1,5)

# ==== define the paramaters for the gradiant descent ==== #
step_size_criteria = 0.01
learning_rate = 0.1

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
    # print('w=',w)
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
    # print('pre=',output)
    return x1,x2,y1,y2,output

# +++++++++++++++++++++++++++ START the gradiant descent to calculate the optimized b3, w3, w4 +++++++++++++++++++++++++++ #

# ==== define the function to calculate step size and new value for the gradiant descent ==== #
def gradiant_descent_value(derivative, learning_rate, value):
    step_size = sum(derivative) * learning_rate
    value -= step_size
    value = round(value,3)
    print('step_size=',step_size)
    return value

# ==== define the function to calculate values need to be optimized ==== #
def optimize_b3_w3_w4(input,b3,w3,w4):
    for i in range(1000):
    # while step_size > step_size_criteria:
        output = blackbox(input, 3.34,-1.22,w3,w4,-1.43,0.57,b3)
        predicted = output[4]
        y1 = softPlus(output[0])
        y2 = softPlus(output[1])
        # print('y1=',y1)
        d_ssr_b3 = [-2 * (observed[i] - predicted[i]) for i in range(len(input))]
        d_ssr_w3 = [-2 *  (observed[i] - predicted[i]) * y1[i] for i in range(len(input))]
        d_ssr_w4 = [-2 *  (observed[i] - predicted[i]) * y2[i] for i in range(len(input))]
        # print('d_ssr=',sum(d_ssr_w3),sum(d_ssr_w4),sum(d_ssr_b3))
        
        b3 = gradiant_descent_value(d_ssr_b3,learning_rate,b3)
        w3 = gradiant_descent_value(d_ssr_w3,learning_rate,w3)
        w4 = gradiant_descent_value(d_ssr_w4,learning_rate,w4)
        print()
        # print('new',w3,w4,b3)
        
    return b3,w3,w4
# ++++++++++++++++++++++++++++++++++++++++++++++++++ END the black box ++++++++++++++++++++++++++++++++++++++++++++++++++ #

result = optimize_b3_w3_w4(input,b3,w3,w4)
print(result)
b3 = result[0]
w3 = result[1]
w4 = result[2]


output_testing = blackbox(input_testing, 3.34,-1.22,w3,w4,-1.43,0.57,b3)[4]

fig, ax = plt.subplots()

plt.scatter(input,observed)

line, = ax.plot(input_testing, output_testing)

plt.show()