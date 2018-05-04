# Code from Tutorial
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "iris-data.txt"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print("\nDimensions of the dataset (rows, columns): " + str(dataset.shape) + "\n")
# head
print("\nPeek at the data itself: \n\n" + str(dataset.head(5)) + "\n")
# descriptions
print("\nStatistical summary of all attributes: \n\n" + str(dataset.describe()))
# class distribution
print("\nBreakdown of the data by the class variable: \n\n" + str(dataset.groupby('class').size()) + "\n")

# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# # histograms
# dataset.hist()
# plt.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric (10-fold cross validation to estimate accuracy)
seed = 7
scoring = 'accuracy'

# Let’s evaluate 6 different algorithms:

# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN).
# Classification and Regression Trees (CART).
# Gaussian Naive Bayes (NB).
# Support Vector Machines (SVM).

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print()

# We can see that it looks like KNN has the largest estimated accuracy score.

# We can also create a plot of the model evaluation results and compare the 
# spread and the mean accuracy of each model. There is a population of accuracy measures 
# for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# We can run the KNN model directly on the validation set and summarize the results 
# as a final accuracy score, a confusion matrix and a classification report.

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. 
# Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support 
# showing excellent results (granted the validation dataset was small).

print()