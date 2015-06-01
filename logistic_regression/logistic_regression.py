# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def get_data():
	X = []
	y = []
	fr = open('data/ex2data1.txt')
	for x in fr:
		data = map(float, x.strip().split(','))
		X.append(data[0:2])
		y.append([data[2]])
	return (np.asarray(X), np.asarray(y))

X, y = get_data()

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def func_h(theta, X):
	temp = sigmoid(X.dot(theta))
	return temp

def error_func(theta, X, y):
	return np.sum(((y) * np.log(func_h(theta, X)) + (1-y) * np.log(1-func_h(theta, X)) ), axis=1)

def cost_func(theta, X, y):
	return -np.mean(error_func(theta, X, y), axis=0)

def gradient(theta, X, y):
	return X.T.dot(func_h(theta, X) - y)

def feature_normalization(data, type='standardization', param = None):
	u"""
		data:
			an numpy array
		type:
			(standardization, min-max)
		param {default None}: 
			dictionary
			if param is provided it is used as mu and sigma when type=standardization else Xmax, Xmin when type=min-max
			rather then calculating those paramanter

		two type of normalization 
		1) standardization or (Z-score normalization)
			is that the features will be rescaled so that they'll have the properties of a standard normal distribution with
				μ = 0 and σ = 1
			where μ is the mean (average) and σ is the standard deviation from the mean
				Z = (X - μ)/σ

			return:
				Z, μ, σ
		2) min-max normalization
			the data is scaled to a fixed range - usually 0 to 1.
			The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard 
			deviations, which can suppress the effect of outliers.

			A Min-Max scaling is typically done via the following equation:
				Z = (X - Xmin)/(Xmax-Xmin)
			return Z, Xmax, Xmin

	"""
	if type == 'standardization':
		if param is None:
			mu = np.mean(data, axis=0)
			sigma =  np.std(data, axis=0)
		else:
			mu = param['mu']
			sigma = param['sigma']
		Z = (data - mu)/sigma
		return Z, mu, sigma

	elif type == 'min-max':
		if param is None:
			Xmin = np.min(data, axis=0)
			Xmax = np.max(data, axis=0)
		else:
			Xmin = param['Xmin']
			Xmax = param['Xmax']

		Xmax = Xmax.astype('float')
		Xmin = Xmin.astype('float')
		Z = (data - Xmin)/(Xmax - Xmin)
		return Z, Xmax, Xmin

def gradient_check(theta, X, y, epsilon):
	return (cost_func(theta + epsilon, X, y) - cost_func(theta - epsilon, X, y))/float(2*epsilon)

X, y = get_data()

X_norm, mu, sigma = feature_normalization(X)

#change X to inculde bias (Add a column of ones to x)
X_one = np.concatenate((np.ones((X_norm.shape[0], 1)), X_norm), axis=1)


epoch = 400
alpha = 0.5
m, n = X_one.shape

#define theta intialize to random
# theta = np.random.random((n, 1))
theta = np.zeros((n, 1))

#overall/total average cost
cost = cost_func(theta, X_one, y)

print cost, '<- intial cost'

epsilon = 10**-6
#update theta
for _ in xrange(epoch):
	
	delta = 1.0/m * gradient(theta, X_one, y)
	theta += -alpha * delta 

	#cost from new theta
	#overall/total average cost
	cost = cost_func(theta, X_one, y)

	print cost

param = {
	'mu' : mu,
	'sigma': sigma
}

p1, _, _ = feature_normalization(np.asarray([34.62365962451697, 78.0246928153624]), param=param)
p2, _, _ = feature_normalization(np.asarray([60.18259938620976, 86.30855209546826]), param=param)

predict1 = func_h(theta, np.concatenate(([1], p1), axis=1))
predict2 = func_h(theta, np.concatenate(([1], p2), axis=1))

print predict1, predict2

points = X[(y == 0).ravel()]
plt.plot(points[:, 0], points[:, 1], 'yo')
points = X[(y == 1).ravel()]
plt.plot(points[:, 0], points[:, 1], 'g*')

x_min = np.min(X[:, 0])
y_min = np.min(X[:, 1])
x_max = np.max(X[:, 0])
y_max = np.max(X[:, 1])
h_x = (x_max - x_min)/500.0
h_y = (y_max - y_min)/500.0
xx, yy = np.meshgrid(np.arange(x_min, x_max+h_x, h_x),
                     np.arange(y_min, y_max+h_y, h_y))

temp = np.asarray(np.c_[xx.ravel(), yy.ravel()], dtype="float32")

p1, _, _ = feature_normalization(np.asarray(temp), param=param)

p1 = func_h(theta, np.concatenate((np.ones((p1.shape[0], 1)), p1), axis=1))
p1 = p1.reshape(xx.shape)
p1[p1<0.5] = 0
p1[p1>=0.5] = 1
plt.contourf(xx, yy, p1, cmap=plt.cm.Paired, alpha=0.8)
plt.show()