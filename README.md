# Iris Dataset Classifcation Machine Learning Program in Python
## "Hello World" of the Machine Learning Era :)

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
![Screenshot](https://github.com/dheerajtuteja/tableau_UKBank/blob/master/Capture1.png)

*descriptions*
```
print(dataset.describe())
```
*class distribution*
```
print(dataset.groupby('class').size())
```
> NOTE: class is the classification column (extreme right) in the dataset


**Data visualizations**

*box and whisker plots*
```
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```

*histograms*
```
dataset.hist()
pyplot.show()
```

*scatter plot matrix*
```
scatter_matrix(dataset)
pyplot.show()
```
