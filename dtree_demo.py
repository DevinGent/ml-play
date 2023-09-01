import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
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
# Let's start with predicting the third value of y_test.
print(y_test.head(5))
print("The third value of y_test is",y_test.values[2])
print("Our model predicts class",dtree.predict(X_test[2].reshape(1,-1)))

predicted = dtree.predict(X_test)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1,2])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Class 0', 'Class 1', 'Class 2'])
cm_display.plot()
print("The accuracy of this model on the test data is {}%.".format(100*metrics.accuracy_score(y_test, predicted)))
plt.savefig('images\confusion-matrix.png')
plt.show() 
# We will compare this versus computing manually.
print("Predicted classes:")
print(predicted)
print("Actual classes:")
print(y_test.values)

correct0=0
correct1=0
correct2=0
i=0

for (predict,actual) in zip(predicted,y_test):
    if predict==actual:
        if predict==0:
            correct0=correct0+1
        elif predict==1:
            correct1=correct1+1
        elif predict==2:
            correct2=correct2+1
    else:
        print("The prediction in row",i)
    i=i+1

print("The number of correct predictions of class 0 was",correct0)
print("The number of correct predictions of class 1 was",correct1)
print("The number of correct predictions of class 2 was",correct2)
# This matches the confusion matrix perfectly!