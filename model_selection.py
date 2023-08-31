import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

np.random.seed(5)

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
########################################################################################
# Let us make a dictionary of the form: ('Model Name': Classifier)
models={}
# Note that for two of the classifiers issues occured later because of reaching the maxium iterations.
# For now we have just set the iterations higher for those classifiers.

classifier_name='Decision Tree'
classifier=tree.DecisionTreeClassifier()
models[classifier_name]=classifier

classifier_name='Random Forest'
classifier=RandomForestClassifier()
models[classifier_name]=classifier

classifier_name='5 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=5)
models[classifier_name]=classifier

classifier_name='8 Neighbors'
classifier=KNeighborsClassifier(n_neighbors=8)
models[classifier_name]=classifier

classifier_name='Logistic Regression'
classifier=LogisticRegression(max_iter=500)
models[classifier_name]=classifier

classifier_name='Linear Discriminant Analysis'
classifier=LinearDiscriminantAnalysis()
models[classifier_name]=classifier

classifier_name='Gaussian Naive Bayes'
classifier=GaussianNB()
models[classifier_name]=classifier

classifier_name='Linear SVC'
classifier=SVC(kernel='linear')
models[classifier_name]=classifier

classifier_name='rbf SVC'
classifier=SVC(kernel='rbf')
models[classifier_name]=classifier

classifier_name='Ada Boost'
classifier=AdaBoostClassifier()
models[classifier_name]=classifier

classifier_name='Neural Net'
classifier=MLPClassifier(max_iter=1000)
models[classifier_name]=classifier

classifier_name='XGBoost'
classifier=XGBClassifier()
models[classifier_name]=classifier

print(models.keys())

# We will also make a dictionary with model names and their predictions for the y_test set.
predictions={}
for key in models:
    models[key].fit(X_train,y_train)
    predictions[key] = models[key].predict(X_test)

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


predict_accuracy=[]
for key in predictions:    
    c_matrix = metrics.confusion_matrix(y_test, predictions[key], labels=[0,1,2])
    # Here labels should include the elements we are inputting and their order should match the display labels below.
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
    cm_display.plot()
    plt.gcf().suptitle(key+' model')
    print("The accuracy of the {} model on the test data is {}%.".format(key,100*metrics.accuracy_score(y_test, predictions[key])))
    predict_accuracy.append(metrics.accuracy_score(y_test, predictions[key]))
    plt.show() 

accuracy_df["Prediction Accuracy"]=predict_accuracy
print(accuracy_df)