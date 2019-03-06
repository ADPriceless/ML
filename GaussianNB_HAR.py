'''
The objective of this practice exercise is to predict current 
human activity based on phisiological activity measurements 
from 53 different features based in the HAR dataset. The training 
and test datasets are provided.
Example is from: https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
'''

import logging
logging.basicConfig(level=logging.INFO)
import requests
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_url = "https://raw.githubusercontent.com/selva86/datasets/master/har_train.csv"
test_url  = "https://raw.githubusercontent.com/selva86/datasets/master/har_validate.csv"

df_train = pd.read_csv(train_url)
X_train = df_train.drop('classe', axis=1)
y_train = df_train['classe']

df_test = pd.read_csv(test_url)
X_test = df_test.drop('classe', axis=1)
y_test = df_test['classe']

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
names = np.unique(y_pred)
sns.heatmap(confusion_mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()