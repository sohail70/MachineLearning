import pickle, gzip
import numpy as np


def load_mnist():
    f = gzip.open("mnist.pkl.gz")
    train1 , train2 , test = pickle.load(f,encoding="latin1")
    train1_x,train1_y = train1
    train2_x,train2_y = train2
    test_x,test_y = test
    train_x = np.vstack((train1_x,train2_x)) 
    train_y = np.append(train1_y,train2_y)

    return (train_x,train_y,test_x,test_y)





def main():
    # Load the dataset
    num_classes = 10
    X_train,y_train,X_test,y_test =  load_mnist()

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

    


if __name__ == '__main__':
    main()