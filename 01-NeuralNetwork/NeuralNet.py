
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




remember --> to many hidden units--> it may over-fit the training data
'''

import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self) -> None:
        # W = [w11 w21;
        #      w12 w22;
        #      w13 w23]
        self.W = np.array([[1,1],[1,1],[1,1]]) 
        # V
        self.V = np.array([[1,1,1]])
        self.biases = np.array([[0,0,0]])

        self.vec_relu = np.vectorize(self.relu) #vectorized relu so that we can input an array of z's
        self.vec_relu_der = np.vectorize(self.relu_derivative)

        self.vec_output_activation = np.vectorize(self.output_activation)
        self.vec_output_activation_der = np.vectorize(self.output_activation_derivative)

        self.learning_rate = 0.001

        self.epochs_to_train = 10
        
        self.training_points = [((2,1),10), ((3,3),21), ((4,5),32),((6,6),42)] # here we didn't consider validation data (which is maybe 20 percents of the training data) --> but if you have one you can add validation accuracy
        self.testing_points = [(1,1),(2,2),(3,3),(5,5),(10,10)]

        self.loss_array = np.array([])
        self.train_accuracy_sum_for_one_epoch = 0
        self.training_accuracy = np.array([])

    def relu(self,z): 
        return np.max([0,z])

    def relu_derivative(self,z):
        if z<0:
            return 0
        else:
            return 1

    def output_activation(self,u): #f(u) = u 
        return u

    def output_activation_derivative(self,u):
        return 1

    def one_step_train(self,x1,x2,y):
        ### Forward Propagation ###
        input_values = np.array([[x1,x2]]) # 2 by 1
        z = np.matmul(self.W , input_values.T) + self.biases.T # 3 by 1 ----> hidden layer aggregation --> array of z1,z2,z3
        a = self.vec_relu(z) # 3 by 1 ----> hidden layer activation ----> array of a1,a2,a3
        u = np.matmul(self.V , a) # ---> output layer aggregation --> u1
        o = self.vec_output_activation(u) # ---> output layer activation ---> o1

        self.loss_array = np.append(self.loss_array , 0.5*(y-o)**2)
        self.train_accuracy_sum_for_one_epoch = self.train_accuracy_sum_for_one_epoch + self.loss_array[-1] ** 2
        ### Gradients ###
        grad_wrt_W = np.matmul(-(y-o) * 1 * self.V.T * self.vec_relu_der(z) , input_values)
        grad_wrt_V = -(y-o) * a
        grad_wrt_b = -(y-o) * self.V.T * self.vec_relu_der(z) * np.array([[1,1,1]]).T

        ### Gradient Descent ###
        self.W = self.W - self.learning_rate * grad_wrt_W
        self.V = self.V - self.learning_rate * grad_wrt_V.T
        self.biases = self.biases - self.learning_rate * grad_wrt_b.T

    def train(self):
        for epoch in range(self.epochs_to_train): # Each epoch is a complete pass through the training dataset
            for x,y in self.training_points:
                self.one_step_train(x[0],x[1],y)
            self.root_mean_square() 

# Accuracy for continuous case is different than accuracy for classification. in classification we check how many of the output of the first epoch matches the y_true but here we can't
# match because https://stackoverflow.com/questions/50520725/accuracy-of-neural-networks-incase-of-doing-prediction-of-a-continuious-variable so we use means square metric 
# to see how much deviation we have from the y_truth with these new set of learned weights in epoch one
    def root_mean_square(self):
        self.train_accuracy_ave_for_one_epoch = self.train_accuracy_sum_for_one_epoch / len(self.training_points) # average of accuracy of data points for one epoch
        self.root_mean = np.sqrt(self.train_accuracy_ave_for_one_epoch)
        self.training_accuracy = np.append(self.training_accuracy , self.root_mean)
        self.train_accuracy_sum_for_one_epoch = 0


    def predict(self,x1,x2):
        ### Forward Propagation ###
        input_values = np.array([[x1,x2]]) # 2 by 1
        z = np.matmul(self.W , input_values.T) + self.biases.T # 3 by 1 ----> hidden layer aggregation --> array of z1,z2,z3
        a = self.vec_relu(z) # 3 by 1 ----> hidden layer activation ----> array of a1,a2,a3
        u = np.matmul(self.V , a) # ---> output layer aggregation --> u1
        o = self.vec_output_activation(u) # ---> output layer activation ---> o1

        return o.item()

    def test_neural_network(self):
        for point in self.testing_points:
            print("Point",point , "Prediction ",self.predict(point[0],point[1]))
            if abs(self.predict(point[0],point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point" , point[0],point[1], " failed to be predicted correctly.")
                return

    def __str__(self):
        return f"W:{self.W} , V:{self.V} , b:{self.biases}"


    def run(self):
        self.train()
        self.test_neural_network()

    

    def loss_info(self):
        ax = plt.figure()
        plt.plot([i for i in range(len(self.loss_array))] , self.loss_array)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("Losses vs iteration over the course of training")
        plt.show()
    

    
    def graphics_with_line(self):
        # Train Data
        x_train = [self.training_points[i][0] for i in range(len(self.training_points))]

        y_truth = [self.training_points[i][1] for i in range(len(self.training_points))]
        y_pred = [self.predict(x_train[i][0],x_train[i][1]) for i in range(len(self.training_points))]
        ax = plt.figure().add_subplot(projection='3d')
    
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        p1 = [ax.plot( [x_train[i][0],x_train[i+1][0]]   ,   [x_train[i][1],x_train[i+1][1]],    [y_truth[i],y_truth[i+1]],'ro--' , label = 'Truth') for i in range(len(self.training_points)-1)]
        p2 = [ax.plot( [x_train[i][0],x_train[i+1][0]]   ,   [x_train[i][1],x_train[i+1][1]],    [y_pred[i],y_pred[i+1]],'bo--' , label='Pred') for i in range(len(self.training_points)-1)]
        


        #handles,labels = plt.gca().get_legend_handles_labels()
        plt.legend()
        plt.show()

    
    def train_graphics(self):
        # Train Data
        x_train = [self.training_points[i][0] for i in range(len(self.training_points))]
        x0 = list(zip(*x_train))[0]
        x1 = list(zip(*x_train))[1]

        y_truth = [self.training_points[i][1] for i in range(len(self.training_points))]
        y_pred = [self.predict(x0[i],x1[i]) for i in range(len(self.training_points))]
        ax = plt.figure(figsize=(10,10)).add_subplot(2,1,1, projection='3d')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        plt.scatter(x0,x1,y_truth , marker='o' ,label = 'Truth' , linewidths=3 , edgecolors='r')
        plt.scatter(x0,x1,y_pred , marker='^' ,label = 'Pred' ,linewidths=3 , edgecolors='b')

        plt.title("Neural Net Prediction of the Training Data") #this is not necessary but just for checking out the fitting process
        plt.legend()

        # Training Accuracy ---> Not exactly accuracy but its MSE --> because accuracy must go up and it has inverse relation with loss but mse has the direct relationship with loss
        plt.subplot(212)
        plt.xlabel("Epoch")
        plt.ylabel("Root mean square error")
        plt.plot(self.training_accuracy)

        plt.show()

    def test_graphics(self):

        x0 = list(zip(*self.testing_points))[0]
        x1 = list(zip(*self.testing_points))[1]
        y_pred = [self.predict(x0[i],x1[i]) for i in range(len(self.testing_points))]
        
        ax = plt.figure().add_subplot(projection='3d')
    
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        
        plt.scatter(x0,x1,y_pred , marker='^' ,label = 'Pred' ,linewidths = 3 , edgecolors = 'b')
        plt.title("Neural Net prediction of the test data")
        plt.legend()
        plt.show()


if __name__== "__main__":
    projectName = "Neural Net"
    print(f"Project: {projectName }")
    myNet = NeuralNetwork()
    myNet.run()
    myNet.train_graphics()
    myNet.test_graphics()
    myNet.loss_info()
    
    print(myNet.__str__())
    