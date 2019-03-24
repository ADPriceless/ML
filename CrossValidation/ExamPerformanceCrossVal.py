'''
In this file, an example dataset is used to learn different data analysis methods.
Dataset from: https://www.kaggle.com/spscientist/students-performance-in-exams/version/1
'''

import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import style
style.use("ggplot")

df = pd.read_csv("students-performance-in-exams/StudentsPerformance.csv")

# Define fold boundaries
NUM_OF_FOLDS = 5
DATA_SIZE = df["gender"].count()
FOLD_SIZE = int(DATA_SIZE / NUM_OF_FOLDS)
lower_fold_bounds = np.arange(0, DATA_SIZE, FOLD_SIZE, dtype=int)
upper_fold_bounds = np.arange(FOLD_SIZE, DATA_SIZE+1, FOLD_SIZE, dtype=int)

# Get relevant data (subject scores, race of parents presumably has no affect on gender of student)
X = np.array(df.loc[:, "math score":"writing score"])
y = np.where(df["gender"]=="female", 1, -1) # female = 1, male = -1

# Split data into folds
X_folds = []
y_folds = []
for lb, ub in zip(lower_fold_bounds, upper_fold_bounds):
    X_folds.append(X[lb:ub, :])
    y_folds.append(y[lb:ub])    # I imagine that this isn't very efficient...

X_train = np.zeros([DATA_SIZE-FOLD_SIZE, X[0].size])
y_train = np.zeros([DATA_SIZE-FOLD_SIZE])
fold_numbers = np.arange(NUM_OF_FOLDS)
train_lb = np.arange(0, X_train[:,0].size, FOLD_SIZE, dtype=int)
train_ub = np.arange(FOLD_SIZE, X_train[:,0].size+1, FOLD_SIZE, dtype=int)

# Set up classifier parameters
classifiers = [SVC(gamma="scale", degree=2, kernel="poly"),
    SVC(gamma="scale", kernel="rbf"),
    SVC(kernel="linear")]
kernels = ["Poly", "RBF", "Linear"]
scores = np.zeros([NUM_OF_FOLDS, len(classifiers)])

# Begin cross-validation
for i, (X_fold, y_fold) in enumerate(zip(X_folds, y_folds)):
    print("\nTest Fold:", i)
    X_test = np.array(X_fold)
    y_test = np.array(y_fold)
    
    # Create training set
    train_fold_numbers = np.setdiff1d(fold_numbers, i)
    for j, fold in enumerate(train_fold_numbers):
        X_train[train_lb[j]:train_ub[j]] = X_folds[fold]
        y_train[train_lb[j]:train_ub[j]] = y_folds[fold]        
        assert (X_train[j*FOLD_SIZE][0] == X_folds[fold][0][0]), "X_folds incorrectly assigned to X_train"

    # Train each classifiers on the current training set
    print("Train:", end=" ")
    for k, (clf, kernel) in enumerate(zip(classifiers, kernels)):
        print(kernel, "... ", sep="", end="")
        clf.fit(X_train, y_train)
        scores[i][k] = clf.score(X_test, y_test)
    print("Done!")

# Find best kernel
mean_scores = np.mean(scores, axis=0)
max_score = np.max(mean_scores)
max_index = int(np.where(max_score==mean_scores)[0])
print()
print(kernels[max_index], "is the best classifier with a mean score of %.2f" % max_score)

# Which gender is most likely to score 100% on all tests?
if classifiers[max_index].predict([[100, 100, 100]]) == 1:
    print("Female is most likely to score 100%") 
else:
    print("Male is most likely to score 100%")
