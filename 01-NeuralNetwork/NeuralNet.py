
'''             w11                 v11
               ----------z1|a1 ---------------
             /                                 \
    x1------ \  w12                    v21      \    
              -----------z2|a2  ---------------   u1 -----------> f(u1) = o1
    x2---     \  w13                  v31     /
                ---------z3|a3 ----------------

same for x2 --> i didn't draw it but the weights for the x2 state to those 3 hidden units are w21 and w22 and w23


W =[w11 w12 w13
    w21 w22 w23]
W'X --->  but we are gonna put the w' in the initialization part of the class directly

activation of the output layer is simple f(x) = x which means it doesn't change the u1  so f(u1) = o1 = u1


for calculating the gradient always try to simplify:

    w1           v1
x1 -----> z1|a1 ----> u1|aa1 ---> loss = 1/2 (y_truth - o1)^2

a1 is relu and aa1 is a linear function aa1(u1) = u1


we find the gradient of loss w.r.t w1 and v1 because they are the knob and we use derivate to see how much changing w1 can change the loss and how much changing v1 can change the loss



this might help for computing the gradients:

constant*v1          Relu_der(z1)                                   [w11 w21
constant*v2    *     Relu_der(z2)    *    [x1 x2]   =     grad of    w12 w22
constant*v3          Relu_der(z3)                                    w13 w23]



'''


import numpy as np


class NeuralNetwork:
    def __init__(self) -> None:
        # W = [w11 w21;
        #      w12 w22;
        #      w13 w23]
        self.W = np.array([[1,1],[1,1],[1,1]]) 
        # V = []
        self.V = np.array([[1,1,1]])
        self.biases = np.array([[0,0,0]])

        self.vec_relu = np.vectorize(self.relu) #vectorized relu so that we can input an array of z's
        self.vec_relu_der = np.vectorize(self.relu_derivative)

        self.learning_rate = 0.001


    def relu(self,z): 
        return np.max([0,z])

    def relu_derivative(self,z):
        if z<0:
            return 0
        else:
            return 1

    def output_activation(u): #f(u) = u 
        return u

    def output_activation_derivative(u)
        return 1

    def one_step_train(self,x1,x2,y):
        ### Forward Propagation ###
        input_values = np.array([[x1,x2]]) # 2 by 1
        z = np.matmul(self.W , input_values.T) + self.biases.T # 3 by 1 ----> hidden layer aggregation --> array of z1,z2,z3
        a = self.vec_relu(z) # 3 by 1 ----> hidden layer activation ----> array of a1,a2,a3
        u = np.matmul(self.V , a) # ---> output layer aggregation --> u1
        o = u # ---> output layer activation ---> o1


        # Gradients
        grad_wrt_W = np.matmul(-(y-o) * 1 * self.V.T * self.vec_relu_der(z) , input_values)
        grad_wrt_V = -(y-o) * a
        grad_wrt_b = -(y-o) * self.V.T * self.vec_relu_der(z) * np.array([[1,1,1]]).T

        # Gradient Descent
        self.W = self.W - self.learning_rate * grad_wrt_W
        self.V = self.V - self.learning_rate * grad_wrt_V
        self.biases = self.biases - self.learning_rate * grad_wrt_b

    def train():
        for epoch in range(self.epochs_to_train):


if __name__== "__main__":
    projectName = "Neural Net"
    print(f"Project: {projectName }")
    myNet = NeuralNetwork()
    myNet.train(1,1,3)
    print("nothing")