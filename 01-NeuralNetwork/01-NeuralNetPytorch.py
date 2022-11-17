import pickle, gzip
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
# we have 54000 data points which means we have 54000 images and in each of is an 1D array of 784 numbers which represent the pixels --> you can reshape them to 28*28 to imshow the picture
# we batch the data into 32 groups ---> so we have 1687 batches and each patch has 32 images and each image is a 784 number 1D array --> use len(batch['X']) and len(batch['X'][0]) to see
# we train each batch --> so we train the network with 32 picture at each go and store the accuracy and loss --> remember by one go i mean a for loop for 32 picture and then average the accuracy and losses
def load_mnist():
    f = gzip.open("mnist.pkl.gz")
    train1 , train2 , test = pickle.load(f,encoding="latin1")
    train1_x,train1_y = train1
    train2_x,train2_y = train2
    test_x,test_y = test
    train_x = np.vstack((train1_x,train2_x)) 
    train_y = np.append(train1_y,train2_y)

    return (train_x,train_y,test_x,test_y)


def batching(X , y , batch_size):
    N = int(len(X) / batch_size )* batch_size
    batch = []

    for i in range(0,N,batch_size):
        batch.append({
            'X': torch.tensor(X[i:i+batch_size] , dtype=torch.float32),
            'y': torch.tensor(y[i:i+batch_size] , dtype=torch.long)
        })

    return batch

def compute_accuracy(prediction , y):
    return np.mean(np.equal(prediction.numpy() , y.numpy())) #torch mean doesn't work on boolean values (yet!)

def training(xyDataPatches , model):

    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    loss = nn.CrossEntropyLoss()
    loss_avg_array_epoch = [] #first element is mean of 1687 patches and 2nd element is the mean of the next 1687 patches
    accuracy_avg_epoch = []
    for epoch in range(10):
        loss_array_batch = []
        batch_accuracies = []
        for batch in tqdm(xyDataPatches): #tqdm is just a progress bar
            #print(batch)
            # set x,y
            x, y = batch['X'] , batch['y']

            '''
             use zero_grad for every batch to set the grad to zero or else the grad would accumulate and the 
             grad that you used for correction step will still be used for the next iters and loss would 
             increase which is a bad thing(zero_grad() accumulation existence is due to convenience in RNN)
            '''
            optimizer.zero_grad()

            # output of NN for this batch ---> we batched the data for performance I guess
            output = model(x) # each row is for each image (each batch has 32 images) and the numbers in each row is the representation of the NN prediction of this image and the max of each row is the current prediction
            # store prediction to compare it to y (label) to compute the accuracy 
            prediction = torch.argmax(output , dim =1) 
            # accuracy means how many predictions are in line with the labels --> we have 32 images so compute accuracy for each of them and then take average 
            batch_accuracies.append(compute_accuracy(prediction, y))
            # cross entropy loss computation --> we use backward on this variable to compute the gradients 
            error = loss(output , y)
            # store the losses
            loss_array_batch.append(error.data.item())
            error.backward()
            # use the computed gradients to update the weights of the neural network
            optimizer.step()
        loss_avg_array_epoch.append(np.mean(np.stack(loss_array_batch)))
        accuracy_avg_epoch.append(np.mean(batch_accuracies))
        print(f"Epoch loss : {loss_avg_array_epoch[-1]} , Epoch accuracy: {accuracy_avg_epoch[-1]}" )
    return model,loss_avg_array_epoch,accuracy_avg_epoch

            
        





def main():
    # Load the dataset
    num_classes = 10
    X_train,y_train,X_test,y_test =  load_mnist()
    # Separating X_train into train and validation
    split_index = int(0.9*len(X_train))
    print(0.9* len(X_train))
    X_valid = X_train[split_index:]
    y_valid = y_train[split_index:]

    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # print(f"{len(X_train)} {len(X_valid)} {len(y_train)} {len(y_valid)}")  

    permutation = [i for i in range(len(X_train))]
    np.random.shuffle(permutation)

    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]


    # splitting dataset into batches
    batch_size = 32
    train_batches = batching(X_train , y_train , batch_size)
    valid_batches = batching(X_valid , y_valid , batch_size)
    test_batches = batching(X_test , y_test , batch_size)


    # NN Model
    model = nn.Sequential(
        nn.Linear(784,30), #784 units for 784 pixels and 30 units for hidden layer --> so this is just a feature mapping from 784 pixels to  30 numbers
        nn.ReLU(),
        nn.Linear(30 , 10), # the output layer has 10 units to represent 10 numbers
    )

    # Training a model
    trained_model , loss , accuracy = training(train_batches , model)
    print("Done Training")


    # Test the data with the trained model



if __name__ == '__main__':
    main()