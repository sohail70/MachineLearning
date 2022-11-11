
'''             w11                 v11
               ----------z1|f1 ---------------
             /                                 \
    x1------ \  w12                    v21      \    
              -----------z2|f2  ---------------   u1 -----------> f(u1) = o1
    x2---     \  w13                  v31     /
                ---------z3|f3 ----------------

same for x2 --> i didn't draw it but the weights for the x2 state to those 3 hidden units are w21 and w22 and w23


W =[w11 w12 w13
    w21 w22 w23]
W'X --->  but we are gonna put the w' in the initialization part of the class directly
'''


import numpy as np


class NeuralNetwork:
    def __init__(self) -> None:
        self.input_to_hidden_weights = np.array([[1,1],[1,1],[1,1]]) # W
        self.hidden_to_output_weights = np.array([1,1,1]) # V
        self.biases = np.array([0,0,0])
    
    def relu(x): 
        return np.max([0,x])

    def relu_derivative(x):
        if x<0:
            return 0
        else:
            return 1

if __name__== "__main__":
    projectName = "Neural Net"
    print(f"Project: {projectName }")