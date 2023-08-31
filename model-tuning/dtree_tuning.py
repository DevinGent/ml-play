import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
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
X_test=test_df.drop('target',axis=1).values
# We receive various warnings later if we try to use a dataframe which can be resolved by using .values here.  
# Let's verify what those values look like.
print(train_df.head(5))
print(X_train[0:5])

y_train=train_df['target']
y_test=test_df['target']
y_train.info()


# We will make a decision tree to start, just to see that everything works fine.
dtree = tree.DecisionTreeClassifier(max_depth=3)
dtree = dtree.fit(X_train, y_train)

# Displaying the tree
plt.figure(figsize=(10,6))
tree.plot_tree(dtree, feature_names=[col for col in df.drop('target',axis=1).columns], fontsize=8) 
plt.show()

# Now we will try to predict the y_test values using our tree and the X_test values.
predicted = dtree.predict(X_test)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print("The accuracy of this model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
plt.show() 


# Next we practice using cross validation.
skfold = StratifiedKFold(n_splits = 10)
cv_scores = cross_val_score(dtree, X_train, y_train, cv = skfold, scoring='accuracy')

print("Cross Validation Scores: ", cv_scores)
print("Average CV Score: ", cv_scores.mean())
# cv_scores consists of 10 scores because we chose n_splits=10.

########################################################################################
# Let us compare with a different decision tree in which we do not limit the depth.

deeptree = tree.DecisionTreeClassifier()
deeptree = deeptree.fit(X_train, y_train)

cv_scores = cross_val_score(deeptree, X_train, y_train, cv = skfold, scoring='accuracy')

print("Cross Validation Scores for the second tree: ", cv_scores)
print("Average CV Score for the second tree: ", cv_scores.mean())

# Comparing the cross validation scores, it seems that the deeper tree might work better.  Let us examine it further.

# Displaying the tree
plt.figure(figsize=(12,6))
tree.plot_tree(deeptree, feature_names=[col for col in df.drop('target',axis=1).columns], fontsize=8) 
plt.show()

# Now we will try to predict the y_test values using our tree and the X_test values.
predicted = deeptree.predict(X_test)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print("The accuracy of the second model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
plt.show() 

#######################################################################################################
# We will now experiment with hyperparameter tuning using a gridsearch.
# First we want to initiate a tree and create a collection of parameters to compare.
tuning_tree=tree.DecisionTreeClassifier(random_state=2)

parameters = {
    'criterion' : ["gini", "entropy", "log_loss"],
    'max_depth':[2,4,6,8,10],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1,2,4,8,16]
}

grid_search=GridSearchCV(estimator=tuning_tree, param_grid=parameters, cv=5, n_jobs=-1,verbose=1, scoring = "accuracy")
grid_search.fit(X_train,y_train)
scores=pd.DataFrame(grid_search.cv_results_)
scores.info()
print(scores.head())
print(scores[['params','mean_test_score','rank_test_score']].sort_values('rank_test_score').head(20))
# There are two sets of parameters which are tied for first place.  Let's see how they perform on the test data.
print(scores[['params','rank_test_score']].iloc[20])

print(scores['params'].values[20])
print(type(scores['params'].values[20]))
for param in scores[scores['rank_test_score']==1]['params']:
    tuned_tree=tree.DecisionTreeClassifier(**param)
    tuned_tree.fit(X_train,y_train)
    predicted=tuned_tree.predict(X_test)
    c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
    # Here labels should include the elements we are inputting and their order should match the display labels below.
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
    cm_display.plot()
    print("The accuracy of the model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
    plt.show() 

# We could find the prediction for the d_tree using the best parameters more easily as follows:
best_tree=grid_search.best_estimator_
predicted=best_tree.predict(X_test)
c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print('The accuracy of the "best" model on the test data is {}%.'.format(100*metrics.accuracy_score(y_test, predicted)))
plt.show() 
# The "best" model ends up performing worse on the test data then the more generic models we tried.  
# This looks like the gridsearch ended up with parameters which helped overfit the training data.

# Just to experiment, let us see if changing the cv value on grid_search changes the result.
grid_search2=GridSearchCV(estimator=tuning_tree, param_grid=parameters, cv=10, n_jobs=-1,verbose=1, scoring = "accuracy")
grid_search2.fit(X_train,y_train)
print(pd.DataFrame(grid_search2.cv_results_).nsmallest(5,'rank_test_score'))
best_tree2=grid_search2.best_estimator_
predicted=best_tree2.predict(X_test)
c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print('The accuracy of the "best" model from the second search on the test data is {}%.'.format(100*metrics.accuracy_score(y_test, predicted)))
plt.show() 
# No improvement.