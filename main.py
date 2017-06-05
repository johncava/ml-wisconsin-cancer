from __future__ import division
import pandas as pd 
import numpy as np
from sklearn import svm
import math

def accuracy(error):
    num = 0
    for index in error:
        if index == 0.0:
            num = num + 1
    return num/len(error)

# Data preparation
data = pd.read_csv('cancer.data')
data = data.replace(['?'], -1)
y = data[data.columns[10]]
y = y.replace([2,4],[0,1])
y = y.astype(float)
x = data.drop(data.columns[[0,10]],axis=1)
x = x.astype(float)
x = np.array(x.values)
y = np.array(y.values)

x_train = x[:488]
x_test = x[488:]
y_train = y[:488]
y_test = y[488]

print x_train.shape

# Linear SVM model
classify = svm.LinearSVC()
classify.fit(x_train,y_train)
error = y_test - classify.predict(x_test)
print accuracy(error)

# Linear RBF SVM model
classify = svm.SVC()
classify.fit(x_train,y_train)
error = y_test - classify.predict(x_test)
print accuracy(error)

