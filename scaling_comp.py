import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
# Importing the classifiers.
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# We will compare how various models (using the code from model_selection.py) perform when using scaled vs unscaled data.
np.random.seed(5)

# First we load the dataset.
df=datasets.load_iris(as_frame=True)['frame']
df.info()

# Next we split it into a training and testing portion.
train_df, test_df = train_test_split(df,train_size=.8)
train_df.info()
test_df.info()
print(test_df.head(5))


# We will separate the predictors from the target.
X_train=train_df.drop('target',axis=1).values
X_test=test_df.drop('target',axis=1).values
# We receive various warnings later if we try to use a dataframe which can be resolved by using .values here.  
# Let's verify what those values look like.

y_train=train_df['target']
y_test=test_df['target']
y_train.info()

# Now we make a copy of X_train and X_test which are scaled.

scaler = StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

########################################################################################
# Let us make a dictionary of the form: ('Model Name': Classifier)
models={}
# Note that for two of the classifiers issues occured later because of reaching the maxium iterations.
# For now we have just set the iterations higher for those classifiers.

classifier_name='Decision Tree'
classifier=tree.DecisionTreeClassifier(random_state=1)
models[classifier_name]=classifier

classifier_name='Random Forest'
classifier=RandomForestClassifier(random_state=1)
models[classifier_name]=classifier

classifier_name='5 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=5)
models[classifier_name]=classifier

classifier_name='8 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=8)
models[classifier_name]=classifier

classifier_name='Logistic Regression'
classifier=LogisticRegression(max_iter=500,random_state=1)
models[classifier_name]=classifier

classifier_name='Linear Discriminant Analysis'
classifier=LinearDiscriminantAnalysis()
models[classifier_name]=classifier

classifier_name='Gaussian Naive Bayes'
classifier=GaussianNB()
models[classifier_name]=classifier

classifier_name='Linear SVC'
classifier=SVC(kernel='linear',random_state=1)
models[classifier_name]=classifier

classifier_name='rbf SVC'
classifier=SVC(kernel='rbf',random_state=1)
models[classifier_name]=classifier

classifier_name='Ada Boost'
classifier=AdaBoostClassifier(random_state=1)
models[classifier_name]=classifier

classifier_name='Neural Net'
classifier=MLPClassifier(max_iter=1000,random_state=1)
models[classifier_name]=classifier

classifier_name='XGBoost'
classifier=XGBClassifier(random_state=1)
models[classifier_name]=classifier

print(models.keys())

# We do the same for a set of classifiers for the scaled data.
######################################################################
scaled_models={}
# Note that for two of the classifiers issues occured later because of reaching the maxium iterations.
# For now we have just set the iterations higher for those classifiers.

classifier_name='Decision Tree'
classifier=tree.DecisionTreeClassifier(random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='Random Forest'
classifier=RandomForestClassifier(random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='5 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=5)
scaled_models[classifier_name]=classifier

classifier_name='8 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=8)
scaled_models[classifier_name]=classifier

classifier_name='Logistic Regression'
classifier=LogisticRegression(max_iter=500,random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='Linear Discriminant Analysis'
classifier=LinearDiscriminantAnalysis()
scaled_models[classifier_name]=classifier

classifier_name='Gaussian Naive Bayes'
classifier=GaussianNB()
scaled_models[classifier_name]=classifier

classifier_name='Linear SVC'
classifier=SVC(kernel='linear',random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='rbf SVC'
classifier=SVC(kernel='rbf',random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='Ada Boost'
classifier=AdaBoostClassifier(random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='Neural Net'
classifier=MLPClassifier(max_iter=1000,random_state=1)
scaled_models[classifier_name]=classifier

classifier_name='XGBoost'
classifier=XGBClassifier(random_state=1)
scaled_models[classifier_name]=classifier

#############################################################################################

# We will also make a dictionary with model names and their predictions for the y_test set.
predictions={}
for key in models:
    models[key].fit(X_train,y_train)
    predictions[key] = models[key].predict(X_test)
# Again, we repeat the process while fitting to the scaled data.

scaled_predictions={}
for key in scaled_models:
    scaled_models[key].fit(scaled_X_train,y_train)
    scaled_predictions[key] = scaled_models[key].predict(scaled_X_test)



# We will make a dataframe to log cross validation scores and accuracy scores on the test data.
accuracy_df = pd.DataFrame({'Model':[key for key in models]})
print(accuracy_df)
skfold = StratifiedKFold(n_splits = 10)
cv_accuracy=[]
for key in models:
    cv_scores = cross_val_score(models[key], X_train, y_train, cv = skfold, scoring='accuracy')
    print("For the {} model we have:".format(key))
    print("Cross Validation Scores: ", cv_scores)
    print("Average CV Score: ", cv_scores.mean())
    cv_accuracy.append(cv_scores.mean())
    print()
accuracy_df['Mean CV Score']=cv_accuracy
print(accuracy_df)

# And we repeat for the scaled data.
cv_accuracy=[]
for key in scaled_models:
    cv_scores = cross_val_score(scaled_models[key], scaled_X_train, y_train, cv = skfold, scoring='accuracy')
    print("For the scaled {} model we have:".format(key))
    print("Cross Validation Scores: ", cv_scores)
    print("Average CV Score: ", cv_scores.mean())
    cv_accuracy.append(cv_scores.mean())
    print()
accuracy_df['Mean CV Score (on scaled)']=cv_accuracy
print(accuracy_df)



predict_accuracy=[]
scaled_predict_accuracy=[]
for key in predictions:    
    c_matrix = metrics.confusion_matrix(y_test, predictions[key], labels=[0,1,2])
    scaled_c_matrix = metrics.confusion_matrix(y_test, scaled_predictions[key], labels=[0,1,2])
    # Here labels should include the elements we are inputting and their order should match the display labels below.

    fig, axs = plt.subplots(1, 2, figsize=(12,6), sharey='row')

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, 
                                                display_labels = ['Class 0', 'Class 1', 'Class 2'])
    cm_display.plot(ax=axs[0])
    scaled_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = scaled_c_matrix, 
                                                display_labels = ['Class 0', 'Class 1', 'Class 2'])
    scaled_cm_display.plot(ax=axs[1])
    axs[0].set_title("Unscaled")
    axs[1].set_title("Scaled")
    fig.suptitle(key+' model')
    print("The accuracy of the {} model on the test data is {}%.".format(key,100*metrics.accuracy_score(y_test, predictions[key])))
    predict_accuracy.append(metrics.accuracy_score(y_test, predictions[key]))
    print("The accuracy of the scaled {} model on the test data is {}%.".format(key,100*metrics.accuracy_score(y_test, scaled_predictions[key])))
    scaled_predict_accuracy.append(metrics.accuracy_score(y_test, scaled_predictions[key]))
    plt.show() 

accuracy_df["Prediction Accuracy"]=predict_accuracy
accuracy_df["Prediction Accuracy (on scaled)"]=scaled_predict_accuracy
print(accuracy_df)