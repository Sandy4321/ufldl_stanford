import scipy.io
import numpy as np
from utils import *

def get_expansion(X, degree=1):
	return np.concatenate([X**i for i in xrange(1, degree+1)], axis=1)

def h_fun(X, theta):
	return X.dot(theta)

#mean square error function
def error_func(y_dash, y):
	return np.sum(0.5*(y_dash - y)**2, axis=1)

def gradient(X, y, theta):
	return X.T.dot(h_fun(X, theta) - y)

def cost_fun(error):
	#overall/total average cost
	cost = np.mean(error, axis=0)
	return cost

#load data 
mat = scipy.io.loadmat('data/ex5data1.mat')
X_train, y_train = mat['X'], mat['y']
X_valid, y_valid = mat['Xval'], mat['yval']
X_test, y_test = mat['Xtest'], mat['ytest']

# print X_train, '\n', y_train
# print X_valid, '\n', y_valid
# print X_test, '\n', y_test

X_train, params = feature_normalization(X_train, type='min-max')
X_valid, _ = feature_normalization(X_valid, type='min-max', params = params)
X_test, _ = feature_normalization(X_test, type='min-max', params = params)


# model selection using validation data {test for degree upto 20}
number_epoch = 200
alpha = 0.01
valid_error_list = []
for d in xrange(1, 20+1):
	# print "----------------------------------------------------------------------"
	X_train_d = get_expansion(X_train, degree=d)
	X_valid_d = get_expansion(X_valid, degree=d)
	
	# change X to inculde bias (Add a column of ones to x)
	X_train_d = np.concatenate((np.ones((X_train_d.shape[0], 1)), X_train_d), axis=1)
	X_valid_d = np.concatenate((np.ones((X_valid_d.shape[0], 1)), X_valid_d), axis=1)

	m, n = X_train_d.shape

	# define theta intialize to random
	theta = np.random.random((n, 1))
	# theta = np.zeros((n, 1))

	for epoch in xrange(number_epoch):

		delta = 1.0/m * gradient(X_train_d, y_train, theta)
		theta = theta - alpha * delta 

	error = error_func(h_fun(X_train_d, theta), y_train)
	# print cost_fun(error), 'training cost for degree ->', d

	error = error_func(h_fun(X_valid_d, theta), y_valid)
	cost = cost_fun(error)
	# print cost, 'validation cost for degree ->', d
	valid_error_list.append(cost)


# model selection using validation data {test for degree upto 20}
number_epoch = 500
alpha = 0.01

#choose degree
d = np.argmin(valid_error_list) + 1
print "selected degree is ->", d 

X_train_d = get_expansion(X_train, degree=d)
X_train_d = np.concatenate((np.ones((X_train_d.shape[0], 1)), X_train_d), axis=1)

m, n = X_train_d.shape

# define theta intialize to random
theta = np.random.random((n, 1))

for epoch in xrange(number_epoch):
	delta = 1.0/m * gradient(X_train_d, y_train, theta)
	theta = theta - alpha * delta

X_test_d = get_expansion(X_test, degree=d)
X_test_d = np.concatenate((np.ones((X_test_d.shape[0], 1)), X_test_d), axis=1)

error = error_func(h_fun(X_test_d, theta), y_test)
print cost_fun(error), 'test cost for degree ->', d