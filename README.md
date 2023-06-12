# Learning material is from StatQuest <br>
https://www.youtube.com/watch?v=CqOfi41LfDw
<br> Main Idea pt1. Step 1. <br>
- One hidden Layer
- Two nodes (y1, y2)
- All the weights and bias are optimaized

<br> Main Idea pt2. Step 2. <br>
- One hidden layer with two nodes ( a top node and a bottom node)
- b3 is to be optimized by using the chain rule and gradiant descent
- It takes 9 steps to get the optimized b3 = 2.61, the step size is -0.002

<br> Backpropagation pt1. Step 3. <br>
- b3, w3, w4 is to be optimized by using the chain rule and gradiant descent
- to define the criteria as for when the gradiant descent should stop optimizing paramaters

<br> Backpropagation pt2. Step 4. <br>
- optimize all paramaters
- backpropagation paramater output sequence matters
- with fancy animation to illustrate the learning process of the plot

<br> mymoudule.py <br>
- Used for animation.py
- Go from here 
  - Export the optimzation mymodule as an application which contains: 
    - User friendly input page of the training data (line 10-14 in mymodule )
      - input array [0, 1, 2]
      - output array [0, 1, 2]
      - max_iteration number 
    - Output animation plot


<br> pt3 <br>
- ReLu activetion function


<br> pt4 <br>
- Multiple inputs and outputs
- 3D plots instead of 2D
- platforms instead of lines

<br> pt5 <br>
-  An illustration of the SoftMax and ArgMax functions.
- Optimized the neural network by putting weights, biases, and training data into arrays. 

<br> pt6 <br>
- Backpropagation with multiple inputs
- Cross Entropy as the loss function
- Calcute the derivative of Cross Entropy with respect to weights and biases
- delta2: the derivatives of the outer layer
- delta1: the derivatives of the inner layer (before the activation funciton)

<br> pt7 <br>
- A convolutional Neural Network (CNN) is the convolution of the input image and a regular neural network. It is like the sample preparation before the samples go through the HPLC or other analysis. 

- Convolution is applying a filter (mask) to turn the input image into a desired array suitable for a neural network. 

- The filter is to be optimized as well. 
