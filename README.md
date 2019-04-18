# Iris Dataset Classifcation Machine Learning Program in Python
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
