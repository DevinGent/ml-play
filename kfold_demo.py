import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn import datasets

np.random.seed(1)

# First we load the dataset.
df=datasets.load_iris(as_frame=True)['frame']
df.info()

# Next we split it into a training and testing portion.
train_df, test_df = train_test_split(df,train_size=.8, random_state=3)
train_df.info()
test_df.info()


# We will separate the predictors from the target.
X_train=train_df.drop('target',axis=1).values

y_train=train_df['target']

# We will make a decision tree.
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

# We practice using cross validation.
skfold = StratifiedKFold(n_splits = 10)
cv_scores = cross_val_score(dtree, X_train, y_train, cv = skfold, scoring='accuracy')

print("Cross Validation Scores: ", cv_scores)
print("Average CV Score: ", cv_scores.mean())
print("Number of CV Scores used in Average: ", len(cv_scores))

