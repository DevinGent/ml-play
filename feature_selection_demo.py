import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn import datasets
import xgboost as xgb


np.random.seed(5)

# In this script we will experiment with feature importance and selection in Python for classifiers.  
# We will use the xgboost classifier and the wine dataset from sklearn.datasets.

# First we load the dataset.
df=datasets.load_wine(as_frame=True)['frame']
df.info()
print(df.head())
# Note that the dataset has a large number of predictor factors (12).

# What are the possible target values?
print(df['target'].unique())
print(df['target'].value_counts(sort=False))


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


model=xgb.XGBClassifier()
print("We will verify the type of the classifier for future reference.")
print(type(model))



skfold = StratifiedKFold(n_splits = 10)

cv_scores = cross_val_score(model, X_train, y_train, cv = skfold, scoring='accuracy')
print("For the xgb model we have:")
print("Cross Validation Scores: ", cv_scores)
print("Average CV Score: ", cv_scores.mean())
cv_accuracy=cv_scores.mean()


model.fit(X_train,y_train)

predictions=model.predict(X_test)

   
c_matrix = metrics.confusion_matrix(y_test, predictions, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
plt.gcf().suptitle('xgb model')
print("The accuracy of the xgb model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predictions)))
plt.show() 

# But what are the most important features?  Let us investigate.
print()
importances=model.feature_importances_
print(importances)
plt.figure(figsize=(8,8))
plt.bar(train_df.drop('target',axis=1).columns,importances)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# It would seem that only 3-5 features are most important in the prediction.
# We will examine this more closely.

thresholds=np.sort(importances)

print("Thresholds:",thresholds)
print("Importances:",importances)

# Let us see what is returned by SelectFromModel:
test_selection= SelectFromModel(model,threshold=.15,prefit=True)
print("Using feature selection with a .15 threshold we return the following:")
print(test_selection)
print("Which has data type",type(test_selection))
# We can use this to reduce X to only these factors.  Let us try it out.
test_values=test_selection.transform(X_train)
# Since only factors 6, 9, 11, and 12 scored higher than .15 we would assume only those columns will remain.
print("Before feature selection:")
print(X_train[0:5])
print("After feature selection:")
print(test_values[0:5])
# It worked!

print("We will compare models using different numbers of factors.  We will always use the most important factors.")
# Now we can judge how models would work using different thresholds / numbers of factors.

factors_used=[]
model_accuracy=[]
mean_cv=[]
for threshold in thresholds:
    # For each pass we select some subset of features.
    select=SelectFromModel(model,threshold=threshold,prefit=True)
    selected_X_train=select.transform(X_train)
    # Using only those features we train and evaluate a new model.
    number_of_factors=selected_X_train.shape[1]
    print("Where we use {} factors:".format(number_of_factors))
    selected_model=xgb.XGBClassifier()
    selected_cv_scores = cross_val_score(selected_model, selected_X_train, y_train, cv = skfold, scoring='accuracy')
    print("Cross Validation Scores: ", selected_cv_scores)
    print("Average CV Score: ", selected_cv_scores.mean())
    # Now we judge the accuracy of the model on the test data.
    selected_model.fit(selected_X_train,y_train)
    selected_X_test=select.transform(X_test)
    selected_predictions=selected_model.predict(selected_X_test)
    print("The accuracy of the xgb model using {} features on the test data is {}%.".format(number_of_factors,
                                                                                            100*metrics.accuracy_score(y_test, selected_predictions)))
    factors_used.append(number_of_factors)
    model_accuracy.append(metrics.accuracy_score(y_test, selected_predictions))
    mean_cv.append(selected_cv_scores.mean())
    print()

# Finally we will visualize the difference in accuracy
plt.figure(figsize=(10,6))
plt.bar(x=factors_used,height=model_accuracy)
plt.xlabel("Number of factors used")
plt.ylabel("Prediction accuracy")
plt.ylim(bottom=.75)
plt.tight_layout()
plt.show()

barchart_df=pd.DataFrame({"Number of Factors Used":factors_used,
                          "Prediction Accuracy on Test Data":model_accuracy,
                          "Average CV Score on Training Data": mean_cv})
barchart_df.sort_values(by='Number of Factors Used',inplace=True)
barchart_df.set_index("Number of Factors Used",inplace=True)
barchart_df.plot.bar(figsize=(10,6),ylim=(.6,1))
plt.legend(loc='lower right', framealpha=.9)
plt.show()
# It seems like 4 factors achieves the best overall success.
# Let us just see what the confusion matrix looks like in that particular case.

thresholds=-np.sort(-importances)
threshold=thresholds[3]
print(threshold)
select=SelectFromModel(model,threshold=threshold,prefit=True)
selected_X_train=select.transform(X_train)
print("There are {} factors now.".format(selected_X_train.shape[1]))

# Using only those features we train and evaluate a new model.
selected_model=xgb.XGBClassifier()
selected_model.fit(selected_X_train,y_train)
selected_X_test=select.transform(X_test)
selected_predictions=selected_model.predict(selected_X_test)

c_matrix = metrics.confusion_matrix(y_test, selected_predictions, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
plt.gcf().suptitle('xgb model')
print("The accuracy of the model with feature selection on the test data is {}%.".format(100*metrics.accuracy_score(y_test, selected_predictions)))
plt.show() 