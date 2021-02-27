import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

'''
Once you have completed the implementation of the 3 learning algorithms, you should qualitatively verify your implementations. In main.py we have
included a block of code that you should uncomment. This code loads a 2D dataset from toy_data.txt, and trains your models using  𝑇=10,𝜆=0.2 . 
main.py will compute 
𝜃  and  𝜃0  for each of the learning algorithms that you have written. Then, it will call plot_toy_data to plot the resulting model and boundary.
'''
'''
Since you have implemented three different learning algorithm for linear classifier, it is interesting to investigate which algorithm would actually converge. Please run it with a larger number of iterations  𝑇  to see whether the algorithm would visually converge. You may also check whether the parameter in your theta converge in the first decimal place. Achieving convergence in longer decimal requires longer iterations, but the conclusion should be the same.

Which of the following algorithm will converge on this dataset? (Choose all that apply.):
average perceptron algorithm converges
pegasos algorithm converges
'''
#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

#average perceptron algorithm va pegasos algorithm dar in problem converge mishan (T=10 ,100,200,300,500,600 bezar
#ta bebini adad ha bad ye modat converge mishan) ama alorithm perceptron haminjoori dar hale taghir kardane ba taghire T
T = 10
L = 0.2

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

def plot_toy_results(algo_name, thetas):
    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    print('theta_0 for', algo_name, 'is', str(thetas[1]))
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)



#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------

T = 25
L = 0.01
# dade ha age well seperated nabashan training accuracy kam mishe. fek kunam 100 darsad accuracy faghat dar halati hast ke linearly seperable bashan dadeha
pct_train_accuracy, pct_val_accuracy = \
   p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))


#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------
'''
You finally have your algorithms up and running, and a way to measure performance! But, it's still unclear what values the hyperparameters
like  𝑇  and  𝜆  should have. In this section, you'll tune these hyperparameters to maximize the performance of each model.

One way to tune your hyperparameters for any given Machine Learning algorithm is to perform a grid search over all the possible combinations of values. 
If your hyperparameters can be any real number, you will need to limit the search to some finite set of possible values for each hyperparameter. 
For efficiency reasons, often you might want to tune one individual parameter, keeping all others constant, and then move onto the next one; Compared 
to a full grid search there are many fewer possible combinations to check, and this is what you'll be doing for the questions below.

In main.py uncomment Problem 8 to run the staff-provided tuning algorithm from utils.py. For the purposes of this assignment, 
please try the following values for  𝑇 : [1, 5, 10, 15, 25, 50] and the following values for  𝜆  [0.001, 0.01, 0.1, 1, 10]. 
For pegasos algorithm, first fix  𝜆=0.01  to tune  𝑇 , and then use the best  𝑇  to tune  𝜆

'''
'''
Isnt tuning T just overfitting?
𝑇  is tuned for generalization purposes. When you say best results, I believe you are talking about the training error. It is probably true the algorithm will
continue to improve with larger  𝑇 . However, what we care is 
the test error. If you have your model fit too well with the training set, you might have overfitting issue, which could fail to generalize to the test set.

plot ha neshoon midan ke az ye T be bad validation ha accuracy shoon paeen miad!
'''


data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

'''
After you have chosen your best method (perceptron, average perceptron or Pegasos) and parameters, use this classifier to compute testing accuracy
on the test set.
We have supplied the feature matrix and labels in main.py as test_bow_features and test_labels.
Note: In practice the validation set is used for tuning hyperparameters while a heldout test set is the final benchmark used to compare 
disparate models that have already been tuned. You may notice that your results using a validation set don't always align with those of the test set,
and this is to be expected.
'''


#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

# Your code here
avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,test_bow_features,train_labels,test_labels,T=25,L=0.01)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Testing accuracy for Pegasos:", avg_peg_val_accuracy))



'''
According to the largest weights (i.e. individual  𝑖  values in your vector), you can find out which unigrams were the most impactful ones 
in predicting positive labels.
Uncomment the relevant part in main.py to call utils.most_explanatory_word.
Report the top ten most explanatory word features for positive classification below:
Also experiment with finding unigrams that were the most impactful in predicting negative labels.
'''
#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
'''
all you are doing is looking for the corresponding words to the largest coefs within theta, so theta_0 is irrelevant
dar vaghe bordare theta ye seri az element hash vazn ziadi dare ke motenazer ba ye seri feature hast ke ona ro paeen bedast miarim
'''
thetas_pegasos = p1.pegasos(train_bow_features, train_labels, 25, 0.01)



best_theta =thetas_pegasos[0] #p1.pegasos(train_bow_features, train_labels, 25, 0.01) # Your code here
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10]) # Top 10
print(sorted_word_features[-10:]) # Worst 10
