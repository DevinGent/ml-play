# You only need to copy the bit at the bottom between hash tags.
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn import datasets
from sklearn import tree
#
import pandas as pd
import numpy.random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Let us make a dictionary of the form: ('Model Name': Classifier)
models={}
# Note that for two of the classifiers issues occured later because of reaching the maxium iterations.
# For now we have just set the iterations higher for those classifiers.
seed=30
numpy.random.seed(seed)
classifier_name='Decision Tree'
classifier=tree.DecisionTreeClassifier()
models[classifier_name]=classifier

classifier_name='Random Forest'
classifier=RandomForestClassifier()
models[classifier_name]=classifier

classifier_name='Extra Trees'
classifier=ExtraTreesClassifier()
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

classifier_name="Gradiant Boost"
classifier=GradientBoostingClassifier()
models[classifier_name]=classifier

classifier_name='Neural Net'
classifier=MLPClassifier(max_iter=1000)
models[classifier_name]=classifier

classifier_name='XGBoost'
classifier=XGBClassifier()
models[classifier_name]=classifier

# We will also add some models with generally preferred settings

classifier_name='Tuned GBoost'
classifier=GradientBoostingClassifier(loss='log_loss',
                                      learning_rate=0.1,
                                      n_estimators=500,
                                      max_depth=3,
                                      max_features='log2') 
models[classifier_name]=classifier

classifier_name='Tuned RForrest'
classifier=RandomForestClassifier(n_estimators=500,
                                  max_features=.25,
                                  criterion='entropy') 
models[classifier_name]=classifier

classifier_name='Tuned SVC'
classifier=SVC(C=0.01,
               gamma=0.1,
               kernel='poly',
               degree=3,
               coef0=10.0) 
models[classifier_name]=classifier

classifier_name='Tuned ETrees'
classifier=ExtraTreesClassifier(n_estimators=1000,
                                max_features='log2',
                                criterion='entropy') 
models[classifier_name]=classifier

classifier_name='Tuned LogReg'
classifier=LogisticRegression(C=1.5,
                              penalty='l1',
                              fit_intercept=True)

model_series=pd.Series(models)
model_series.rename('Model', inplace=True)
modelsdf=pd.DataFrame(model_series)
modelsdf.index.name='Model Name'
#
df2=datasets.load_iris(as_frame=True)['frame']
df2.info()

# Next we split it into a training and testing portion.
train_df, test_df = train_test_split(df2,train_size=.8, random_state=3)
train_df.info()
test_df.info()

np.random.seed(6)
# We will separate the predictors from the target.
X=train_df.drop('target',axis=1).values
y=train_df['target']
model=modelsdf['Model'].iloc[0]

skfold = StratifiedKFold(n_splits = 10)
scoring='accuracy'
cross_val_params={'X':X, 
                  'y':y, 
                  'cv' : skfold, 
                  'scoring' : 'accuracy'}
cv_scores = cross_val_score(model, X, y, cv = skfold, scoring='accuracy')
print(cv_scores)
print(len(cv_scores))

dataframe=modelsdf



###################################################################################
###################################################################################
###################################################################################
###################################################################################
from sklearn.model_selection import cross_val_score
import numpy as np
number_of_scores=len(cross_val_score(estimator=dataframe['Model'].iloc[0], **cross_val_params))
frame_length=dataframe.shape[0]
for i in range(number_of_scores):
    dataframe['CV{}'.format(i+1)]=[np.nan]*frame_length

dataframe['CV Mean']=[np.nan]*frame_length
dataframe['CV Std'] = [np.nan]*frame_length
for model in dataframe.index:
    cv_scores=cross_val_score(estimator=dataframe['Model'].loc[model], **cross_val_params)
    for i in range(number_of_scores):
        dataframe['CV{}'.format(i+1)].loc[model]=cv_scores[i]
    dataframe['CV Mean'].loc[model]=cv_scores.mean()
    dataframe['CV Std'].loc[model]=cv_scores.std()
    print(dataframe)

###################################################################################
###################################################################################
###################################################################################
###################################################################################