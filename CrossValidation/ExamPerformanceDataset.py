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

''' Plot some data just for the sake of practicing. '''

df = pd.read_csv("students-performance-in-exams/StudentsPerformance.csv")
# print(df.head())

# Plot mean math scores:
mean_math_score = df["math score"].mean()
fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# By gender
mean_math_scores_by_gender = df[["gender", "math score"]].groupby("gender").mean()
ax1.bar(mean_math_scores_by_gender.index.values, 
    mean_math_scores_by_gender["math score"])
ax1.set_ylabel("Score / %")
ax1.set_xlabel("Gender")
ax1.axhline(y=mean_math_score, color='b')

# By parental level of education (ple)
mean_math_score_by_ple = df[["parental level of education", \
    "math score"]].groupby("parental level of education").mean()
ax2.bar(mean_math_score_by_ple.index.values, 
    mean_math_score_by_ple["math score"],
    bottom=0.25)
ax2.set_xlabel("Parental Level of Education")
plt.xticks(rotation="45")
ax2.axhline(y=mean_math_score, color='b')

# Plot math score histograms 
fig2 = plt.figure(2)
ax3 = fig2.add_subplot(111)
female_df = df[df["gender"] == "female"]
male_df = df[df["gender"] == "male"]
ax3.hist([female_df["math score"], male_df["math score"]])
ax3.set_xlabel("Score / %")
ax3.set_ylabel("Number of students")
ax3.set_xticks(np.arange(0, 110, 10))
ax3.legend(["female", "male"])

# plt.show()

# Find female-to-male ratio (from training data)
s = df.loc[:750, "gender"]
f_to_m_ratio = s[s=="female"].count() / s[s=="male"].count()
print("Female-to-male ratio = %.2f:1" % f_to_m_ratio)

''' See if we can predict a student's gender based on 
their math, reading and writing scores. N.B. Assuming 
that factors such as race are not useful features for 
the classifier to predict students' genders. '''

X = np.array(df.loc[:, "math score":"writing score"])
y = np.where(df["gender"]=="female", 1, -1) # female = 1, male = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train SVC classifiers with different kernels
classifiers = [SVC(gamma="scale", degree=2, kernel="poly"),
    SVC(gamma="scale", kernel="rbf"),
    SVC(kernel="linear")]
kernels = ["Poly", "RBF", "Linear"]
for clf, kernel in zip(classifiers, kernels):
    print(kernel, "Kernel\n-------------")

    clf.fit(X_train, y_train)
    print("Score:", clf.score(X_test, y_test))

    # Which gender is most likely to score 100% on all tests?
    if clf.predict([[100, 100, 100]]) == 1:
        print("Female is most likely to score 100%") 
    else:
        print("Male is most likely to score 100%")
    print()

    # They never seem to agree :(
