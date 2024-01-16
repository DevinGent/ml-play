import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


# We will try to scale in two ways.  First we will get a training and testing set consisting only of values for X and y.

# OPTION 1 #######################################################
# We will separate the predictors from the target.
X_train=train_df.drop('target',axis=1).values
X_test=test_df.drop('target',axis=1).values
y_train=train_df['target']
y_test=test_df['target']

scaler=StandardScaler()
print("X_train before scaling:")
print(X_train)
X_train=scaler.fit_transform(X_train)
print("X_train after scaling:")
print(X_train)

print("X_test before scaling:")
print(X_test)
X_test=scaler.transform(X_test)
print("X_test after scaling:")
print(X_test)

# For this demo we will make a decision tree.
dtree = tree.DecisionTreeClassifier(random_state=1)
dtree = dtree.fit(X_train, y_train)

predicted = dtree.predict(X_test)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print("The accuracy of this model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
plt.show()
scaled_X_train=X_train
scaled_X_test=X_test


# Now we will scale while X train/test are dataframes, not values.
# OPTION 2 ########################################################################################################


X_train=train_df.drop('target',axis=1)
X_test=test_df.drop('target',axis=1)



print("X_train before scaling:")
print(X_train)
print(X_train.describe())
X_train=scaler.fit_transform(X_train)
print("X_train after scaling:")
print(X_train)
# NOTE! The following code fails because scaldar.transform() returns a numpy multidimensional array, not a dataframe.
# print(X_train.describe())
# If we are really curious about the stats we can make it work as follows.
described=pd.DataFrame(X_train, columns=X_test.columns)
print(described.describe())


print("X_test before scaling:")
print(X_test)
X_test=scaler.transform(X_test)
print("X_test after scaling:")
print(X_test)

dtree = dtree.fit(X_train, y_train)

predicted = dtree.predict(X_test)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print("The accuracy of this model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
plt.show()

# Finally let's compare the results of our two methods.
print(scaled_X_train==X_train)
print(scaled_X_test==X_test)
# They match!
