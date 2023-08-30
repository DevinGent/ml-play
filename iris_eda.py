import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns

# First we load the iris dataset from the built in datasets of sklearn.
# Using as_frame=True ensures that the result includes a pandas dataframe with the data.
iris=sklearn.datasets.load_iris(as_frame=True)

# We check what kind of object load_iris produces.
print(type(iris))

# A quick search reveals that a bunch type has many similarities and shared behaviors with a dictionary.  
# Let us look at what the keys are:
print(iris.keys())

# Let's look at 'DESCR' to learn more about the iris dataset.
print(iris['DESCR'])

# Now let us look at the frame to see if it is the sort of dataframe we would expect from the description.
print(iris['frame'])

# It is, so let us set the dataframe itself as df.
df=iris['frame']
df.info()

# Note that the target column, from 'DESCR' should represent classes. We verify the names of the classes.
print(iris['target_names'])
# We also check that feature_names behaves as expected.  
print(iris['feature_names'])

# We know the information is given in centimeters, so we will rename columns to make working with the dataframe easier.
df.rename(columns={'sepal length (cm)':'Sepal Length',
                   'sepal width (cm)':'Sepal Width',
                   'petal length (cm)':'Petal Length',
                   'petal width (cm)':'Petal Width',
                   'target':'Class'},
                   inplace=True)

df.info()

# Finally, we saw that their should be 50 flowers of each class.  Let us verify that.
print(df['Class'].value_counts())
# It matches.

#####################################################################################################
# We will start our exploratory analysis.

# First we will look at box and whisker plots of the data.
print(df.drop('Class',axis=1).describe())

fig, axs=plt.subplots(2,2,figsize=(12,7))

for (col,ax) in zip(df.drop('Class',axis=1),axs.flatten()):
    sns.boxplot(data=[df[col]]+[df[df['Class']==i][col] for i in range(3)], ax=ax)
    # We want to label the x ticks properly.
    ax.set_xticklabels(['All',0,1,2])
    # We also want the x and y labels to be informative
    ax.set_xlabel('Class')
    ax.set_ylabel(col+' (cm)')

plt.tight_layout()
# The graphs would be hard to read, so we add space between them
fig.subplots_adjust(hspace=0.3,wspace=.2)
plt.show()

# We will look at a set of scatterplots to see how data may be related.
axes= pd.plotting.scatter_matrix(df.drop('Class',axis=1), figsize=(12,7))

for ax in axes.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
    ax.tick_params('x', labelrotation=0)

plt.tight_layout()
plt.show()
# Because there are so few datapoints in total (comparatively speaking) the result doesn't tell us too much visually.

# Next let us look at the correlation of the factors.
corr_matrix=df.drop('Class',axis=1).corr()
print(corr_matrix)


# We will visualize this with a heatmap.
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True)
plt.tight_layout()
plt.show()

# Sepal Width has little correlation with any factor but Sepal Length, Petal Length, and Petal Width demonstrate more 
# significant correlation.  Further, Petal Length and Petal Width are closely correlated.



# Let us consider the class breakdown of the 50 flowers with the longest sepal length.

# We practice by try to display the 50 rows of df which have the highest Sepal Length
print("We experiment with sorting the dataframe:")
srted_df= df.sort_values(by='Sepal Length', ascending=False)
print(srted_df)
srted_df=srted_df.head(50)
print(srted_df)
print(srted_df.head(50)['Class'])
print(srted_df.head(50)['Class'].value_counts())
piedata=srted_df.head(50)['Class'].value_counts()
print(piedata)
print(piedata.index.tolist())
pielabels=['Class '+ str(i) for i in piedata.index]
print(pielabels)
print("Having experimented, we can now produce pie charts as desired.")

plt.pie(piedata,labels=pielabels,
        colors = sns.color_palette(), autopct='%.0f%%')
plt.gcf().suptitle('50 Longest Sepal Length')
plt.show()

# We can also do the same for Sepal Width.

srted_df= df.sort_values(by='Sepal Width', ascending=False)
piedata=srted_df.head(50)['Class'].value_counts()
pielabels=['Class '+ str(i) for i in piedata.index]
plt.pie(piedata,labels=pielabels,
        colors = sns.color_palette(), autopct='%.0f%%')
plt.gcf().suptitle('50 Widest Sepal Width')
plt.show()