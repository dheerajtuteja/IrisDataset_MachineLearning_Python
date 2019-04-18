# Iris Dataset Classifcation - Machine Learning Program in Python
## "Hello World" of the "Machine Learning" era :)

**Load libraries**

```
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```

**Load dataset**

```
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
```
> NOTE : "dataset" is the table created for analysis.

**Descriptive statistics**

*shape*
```
print(dataset.shape)
```
> Output : (150,5)

*head*
```
print(dataset.head(20))
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Dataset.PNG)

*descriptions*
```
print(dataset.describe())
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Description.PNG)

*class distribution*
```
print(dataset.groupby('class').size())
```
> NOTE: class is the classification column (extreme right) in the dataset
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Class.PNG)

**Data visualizations**

*box and whisker plots*
```
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Bob%20%26%20Whisker.PNG)

*histograms*
```
dataset.hist()
pyplot.show()
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Histogram.PNG)

*scatter plot matrix*
```
scatter_matrix(dataset)
pyplot.show()
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Scatter%20Plot.PNG)

**Evaluate Algorithms**

*Split-out validation dataset*
```
array = dataset.values
X = array[:,0:4]
```
> Independent variables in the dataset (X)
```
Y = array[:,4]
```
> Dependent variable in the dataset (Y)
```
validation_size = 0.20 # 80% Training vs 20% Test (Validation)
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

**Machine Learning Algorithms Selection / Check**
```
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
```

**Evaluate each Machine Learning model**
```
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
```  
> We will use 10-fold cross-validation to estimate accuracy on unseen data. 
> This will split our dataset into 10 parts, e.g. the model will train on 9 and test on 1 and repeat for all combinations of train-test splits.
``` 
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```
**Select the best Model**
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Model%20Output.PNG)

**Comparison of Machine Learning algorithms**
```
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Algorithm%20Comparison.PNG)

**Make predictions on validation dataset**
```
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
```
> We can see that the accuracy is 0.9 or 90%.
```
print(confusion_matrix(Y_validation, predictions))
```
> The confusion matrix provides an indication of the three errors made.(2 + 1)
```
print(classification_report(Y_validation, predictions))
```
![Screenshot](https://github.com/dheerajtuteja/IrisDataset_MachineLearning_Python/blob/master/Output.PNG)
