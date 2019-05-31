# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold  #EDIT: I had to import KFold 
from sklearn import datasets
import matplotlib.pyplot as plt
 
# adapt this to run 

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
#see runAllClfs function
 
# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparameter settings
#added LogisticRegression

# 3. find some simple data
#imported sklearn datasets and used iris dataset.

# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings
#see createPlots method

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
#createPlots method will also save plots to user's working directory.

# 6. Investigate grid search function


#EDIT: array M includes the X's
#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])

#EDIT: array L includes the Y's, they're all ones and as such is only for example (an ML algorithm would always predict 1).
#L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])

#we will use the classic iris dataset as our simple data for classification.
iris = datasets.load_iris()
M = iris.data
L = iris.target
#EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)

#EDIT: Let's see what we have.
print(data)


# data expanded
M, L, n_folds = data
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=n_folds)

print(kf)

#if you want to see all the values in NumPy arrays.
#np.set_printoptions(threshold=np.inf)
#EDIT: Show what is kf.split doing
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    print("k fold = ", ids)
    print("            train indexes", train_index)
    print("            test indexes", test_index)

#EDIT: A function, "run", to run all our classifiers against our data.

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    clf = a_clf(**clf_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
                                                                             #      for those corresponding to a formal parameter
                                                                             #      in a dictionary.
            
    clf.fit(M[train_index], L[train_index])   #EDIT: First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = clf.predict(M[test_index])         #EDIT: Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret

#Use run function
results = run(RandomForestClassifier, data, clf_hyper={})

print(results)

def runAllClfs(clfsAndHypers, data):
    allResults = []
    for i in clfs:
        for k, v in i.items():
            results = run(k, data, v)
            allResults.append(results)
            print(results)
    return allResults

def createAccuracyDict(results):
    accuracyDict = {}
    for result in results:
        for key in result:
            k1 = result[key]['clf']
            v1 = result[key]['accuracy']
            k1Test = str(k1)
            k1Test = k1Test.replace('            ',' ') # remove large spaces from string
            k1Test = k1Test.replace('          ',' ')
        
            if k1Test in accuracyDict:
                accuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
            else:
                accuracyDict[k1Test] = [v1]
    return accuracyDict

#the code for this method is pulled directly from Chris's office hours example.
def createPlots(accuracyDict):
    # for determining maximum frequency (# of kfolds) for histogram y-axis
    n = max(len(v1) for k1, v1 in accuracyDict.items())
    
    # for naming the plots
    filename_prefix = 'clf_Histograms_'
    
    # initialize the plot_num counter for incrementing in the loop below
    plot_num = 1 
    
    # Adjust matplotlib subplots for easy terminal window viewing
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.6      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for space between subplots,
                   # expressed as a fraction of the average axis width
    hspace = 0.2   # the amount of height reserved for space between subplots,
                   # expressed as a fraction of the average axis height
    
    #create the histograms
    #matplotlib is used to create the histograms: https://matplotlib.org/index.html
    for k1, v1 in accuracyDict.items():
        # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
        fig = plt.figure(figsize =(10,10)) # This dictates the size of our histograms
        ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
        plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
        ax.set_title(k1, fontsize=25) # increase title fontsize for readability
        ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
        ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
        ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
        ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
        ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
        #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.
    
        # pass in subplot adjustments from above.
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plot_num_str = str(plot_num) #convert plot number to string
        filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
        plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
        plot_num = plot_num+1 # increment the plot_num counter by 1
    plt.show()

clfs = [];
clfs.append({RandomForestClassifier: {'n_estimators': 200, 'max_depth': 3, 'random_state': 10}})
clfs.append({RandomForestClassifier: {'n_estimators': 250, 'max_depth': 2, 'random_state': 10}})
clfs.append({RandomForestClassifier: {'n_estimators': 300, 'max_depth': 1, 'random_state': 10}})
clfs.append({LogisticRegression: {'max_iter': 150, 'n_jobs': 2, 'random_state': 12}})
clfs.append({LogisticRegression: {'max_iter': 300, 'n_jobs': 2, 'random_state': 12}})
allResults = runAllClfs(clfs, data)
accuracyDict = createAccuracyDict(allResults)
createPlots(accuracyDict)

#After explaining.... talk about lists and dictionaries.
#https://docs.python.org/3/tutorial/

#Also... Here's your clfs 
#https://scikit-learn.org/stable/supervised_learning.html

#Go through examples in this order:
# ** operator 
# list1, 
# dictionary, 
# list2