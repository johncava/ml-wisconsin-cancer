from __future__ import division
import pandas as pd 
import numpy as np
from sklearn import svm
import math
import torch
from torch.autograd import Variable

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
svm_acc = accuracy(error) * 100

# Linear RBF SVM model
classify = svm.SVC()
classify.fit(x_train,y_train)
error = y_test - classify.predict(x_test)
svm_rbf_acc = accuracy(error) * 100

# Deep Learning Model
inpt_train_x = torch.from_numpy(x_train)
inpt_train_x = inpt_train_x.float()
inpt_train_y = torch.from_numpy(y_train)
inpt_train_y = inpt_train_y.float()

inpt_train_x = Variable(inpt_train_x)
inpt_train_y = Variable(inpt_train_y, requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(9, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

for t in range(10000):

    y_pred = model(inpt_train_x)

    loss = loss_fn(y_pred, inpt_train_y)
    if t%100 == 0:
        print(t, loss.data[0])

    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad.data


inpt_test_x = torch.from_numpy(x_train)
inpt_test_x = inpt_test_x.float()
inpt_test_x = Variable(inpt_test_x)

array = model.forward(inpt_test_x).data.numpy()
new_array = []
for index in array:
    if index[0] > 0.5:
        new_array.append(1)
    else:
        new_array.append(0)
new_array = np.array(new_array)
error = y_test - new_array

deep_acc = accuracy(error) *100

print 'SVM Accuracy: ', svm_acc
print 'SVM RBF Accuracy: ', svm_rbf_acc
print 'Neural Network Accuracy: ', deep_acc
