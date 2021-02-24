from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    
    #raise NotImplementedError
    return max(0,1-(label*(np.dot(theta,feature_vector)+theta_0)))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    #raise NotImplementedError
    
    ''' #first solution
    loss = 0
    for i in range(feature_matrix.shape[0]): 
        loss += hinge_loss_single(feature_matrix[i],labels[i],theta,theta_0)
        #return loss/(i+1)
    '''
    #second sol: maybe you could use np.vectorize instead of for loop
    
    #third solution
    loss = []
    for i in range(feature_matrix.shape[0]): 
        loss.append( hinge_loss_single(feature_matrix[i],labels[i],theta,theta_0))


    return np.average(loss)


'''
Now you will implement the single step update for the perceptron algorithm (implemented with  0−1  loss). You will be given the feature vector as an array of numbers, the current  𝜃  and  𝜃0  parameters, and the correct label of the feature vector. The function should return a tuple in which the first element is the correctly updated value of  𝜃  and the second element is the correctly updated value of  𝜃0 .

Available Functions: You have access to the NumPy python library as np.

Tip:: Because of numerical instabilities, it is preferable to identify  0  with a small range  [−𝜀,𝜀] . That is, when  𝑥  is a float, “ 𝑥=0 " should be checked with  |𝑥|<𝜀 .

'''
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    #raise NotImplementedError
    if label*(np.dot(feature_vector,current_theta)+ current_theta_0) <= 10**(-6) :
        current_theta =np.array( current_theta + label*feature_vector)
        current_theta_0 = current_theta_0 + label
    return current_theta,current_theta_0 #parantez ham nazari tuple hast farghi nadare

'''
In this step you will implement the full perceptron algorithm. You will be given the same feature matrix and labels array as you were given in The Complete Hinge Loss. You will also be given  𝑇 , the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. Initialize  𝜃  and  𝜃0  to zero. This function should return a tuple in which the first element is the final value of  𝜃  and the second element is the value of  𝜃0 .

Tip: Call the function perceptron_single_step_update directly without coding it again.

Hint: Make sure you initialize theta to a 1D array of shape (n,) and not a 2D array of shape (1, n).

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration. The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

Available Functions: You have access to the NumPy python library as np and perceptron_single_step_update which you have already implemented.

'''
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta = np.zeros([feature_matrix.shape[1]]) #ye table ro tasavor kun ke row ha mishan dadeha va har ro n ta column dare ke feature har dade hast
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            #pass
            theta,theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)

    #raise NotImplementedError
    return theta,theta_0


'''
The average perceptron will add a modification to the original perceptron algorithm: since the basic algorithm continues updating as the algorithm runs, nudging parameters in possibly conflicting directions, it is better to take an average of those parameters as the final answer. Every update of the algorithm is the same as before. The returned parameters  𝜃 , however, are an average of the  𝜃 s across the  𝑛𝑇  steps:

𝜃𝑓𝑖𝑛𝑎𝑙=(1/𝑛𝑇) (𝜃(1)+𝜃(2)+...+𝜃(𝑛𝑇)) 

You will now implement the average perceptron algorithm. This function should be constructed similarly to the Full Perceptron Algorithm above, except that it should return the average values of  𝜃  and  𝜃0 

Tip: Tracking a moving average through loops is difficult, but tracking a sum through loops is simple.

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration. The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

Available Functions: You have access to the NumPy python library as np and perceptron_single_step_update which you have already implemented.

'''


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    #raise NotImplementedError
    theta_sum = np.zeros([feature_matrix.shape[1]])
    theta_0_sum = 0
    theta = np.zeros([feature_matrix.shape[1]]) #ye table ro tasavor kun ke row ha mishan dadeha va har ro n ta column dare ke feature har dade hast
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            #pass
            theta,theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return theta_sum/(T*feature_matrix.shape[0]) , theta_0_sum/(T*feature_matrix.shape[0])


'''
Now you will implement the Pegasos algorithm. For more information, refer to the original paper at original paper (Pegasos: Primal Estimated sub-GrAdient
SOlver for SVM)

The following pseudo-code describes the Pegasos update rule.

Pegasos update rule (𝑥(𝑖),𝑦(𝑖),𝜆,𝜂,𝜃): 
if  𝑦(𝑖)(𝜃⋅𝑥(𝑖))≤1  then
  update  𝜃=(1−𝜂𝜆)𝜃+𝜂𝑦(𝑖)𝑥(𝑖) 
else:
  update  𝜃=(1−𝜂𝜆)𝜃 

The  𝜂  parameter is a decaying factor that will decrease over time. The  𝜆  parameter is a regularizing parameter.

In this problem, you will need to adapt this update rule to add a bias term ( 𝜃0 ) to the hypothesis, but take care not to penalize the magnitude of  𝜃0 . 
why? The SVM regularization tries to minimize the norm of theta. The pegasos update rule given above is for theta with regularization. Regularization is not applicable for the linear classifier bias(offset)
term theta_0 ( As regularization term is used to increase the linear classifier margin which is 1/norm(theta)). Hence we should not try to minimize the norm of theta_0 as we do for theta.
'''



'''
Next you will implement the single step update for the Pegasos algorithm. This function is very similar to the function that you implemented in
Perceptron Single Step Update, except that it should utilize the Pegasos parameter update rules instead of those for perceptron.
The function will also be passed a  𝜆  and  𝜂  value to use for updates.
Available Functions: You have access to the NumPy python library as np.
'''



def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    #raise NotImplementedError
    if label*(np.dot(feature_vector,current_theta)+current_theta_0) <= 1:
        current_theta = (1-L*eta)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label*1
    else:
        current_theta = (1-L*eta)*current_theta
        current_theta_0 = current_theta_0
    
    return current_theta,current_theta_0


'''

Finally you will implement the full Pegasos algorithm. You will be given the same feature matrix and labels array as you were given in Full Perceptron Algorithm.
You will also be given  𝑇 , the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. 
Initialize  𝜃  and  𝜃0  to zero. For each update, set  𝜂=1𝑡√  where  𝑡  is a counter for the number of updates performed so far (between  1  and  𝑛𝑇  inclusive).
This function should return a tuple in which the first element is the final value of  𝜃  and the second element is the value of  𝜃0 .

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration.
The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

Available Functions: You have access to the NumPy python library as np and pegasos_single_step_update which you have already implemented.


'''
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    #raise NotImplementedError
    j=1
    theta = np.zeros([feature_matrix.shape[1]])
    theta_0 = 0 
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1/np.sqrt(j)
            j+=1
            #breakpoint()
            theta,theta_0 = pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,theta,theta_0)
    
    return theta,theta_0

# Part II
'''
Now that you have verified the correctness of your implementations, you are ready to tackle the main task of this project: building a classifier 
that labels reviews as positive or negative using text-based features and the linear classifiers that you implemented in the previous section!

The Data
The data consists of several reviews, each of which has been labeled with  −1  or  +1 , corresponding to a negative or positive review, respectively.
The original data has been split into four files:

reviews_train.tsv (4000 examples)
reviews_validation.tsv (500 examples)
reviews_test.tsv (500 examples)

To get a feel for how the data looks, we suggest first opening the files with a text editor, spreadsheet program, or other scientific software package
(like pandas). dastor ine: df = pd.read_csv('~/MITCourse/MITcourse/Project1(sentiment_analysis)/reviews_train.tsv',sep='\t', encoding='cp1252')
ya ba : reviews_train = pd.read_csv("reviews_train.tsv", sep='\t',encoding = 'unicode_escape')
Translating reviews to feature vectors
We will convert review texts into feature vectors using a bag of words approach. We start by compiling all the words that appear in a training set of
reviews into a dictionary , thereby producing a list of  𝑑  unique words.


We can then transform each of the reviews into a feature vector of length  𝑑  by setting the  𝑖th  coordinate of the feature vector to  1  if the  𝑖th 
word in the dictionary appears in the review, or  0  otherwise. For instance, consider two simple documents “Mary loves apples" and “Red apples".
In this case, the dictionary is the set  {Mary;loves;apples;red} , and the documents are represented as  (1;1;1;0)  and  (0;0;1;1) .

A bag of words model can be easily expanded to include phrases of length  𝑚 . A unigram model is the case for which  𝑚=1 . 
In the example, the unigram dictionary would be  (Mary;loves;apples;red) . In the bigram case,  𝑚=2 , the dictionary is 
(Mary loves;loves apples;Red apples) , and representations for each sample are  (1;1;0),(0;0;1) . In this section, you will 
only use the unigram word features. These functions are already implemented for you in the bag of words function.
In utils.py, we have supplied you with the load data function, which can be used to read the .tsv files and returns the labels and texts.
We have also supplied you with the bag_of_words function in project1.py, which takes the raw data and returns dictionary of unigram words.
The resulting dictionary is an input to extract_bow_feature_vectors which computes a feature matrix of ones and zeros that can be used as the input
for the classification algorithms. Using the feature matrix and your implementation of learning algorithms from before, you will be able to compute 
 𝜃  and  𝜃0 .


'''

'''
Implement a classification function that uses  𝜃  and  𝜃0  to classify a set of data points. You are given the feature matrix,  𝜃 , and  𝜃0  as defined in previous sections. This function should return a numpy array of -1s and 1s. If a prediction is greater than zero, it should be considered a positive classification.

Available Functions: You have access to the NumPy python library as np.

Tip:: As in previous exercises, when  𝑥  is a float, “ 𝑥=0 " should be checked with  |𝑥|<𝜖 .
'''


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    raise NotImplementedError


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
