import numpy as np
import matplotlib.pyplot as plt

def get_data(num_label):
	X = []
	y = []
	fr = open('data/ex2data2.txt')
	labels = np.eye(num_label)
	for x in fr:
		data = map(float, x.strip().split(','))
		X.append(data[0:2])
		y.append(labels[data[2]])

	return (np.asarray(X), np.asarray(y))

def softmax(x):
	exp = np.exp(x)
	temp = np.sum(exp, axis=1)
	temp = temp.reshape((temp.shape[0], 1))
	return exp/temp

def func_h(theta, X):
	temp = softmax(X.dot(theta))
	return temp

def error_func(theta, X, y):
	return np.sum(y * np.log(func_h(theta, X)), axis=1)

def cost_func(theta, X, y):
	return -np.mean(error_func(theta, X, y), axis=0)

def gradient(theta, X, y):
	return X.T.dot(func_h(theta, X) - y)

def featureNormalize(X, mu=None, sigma=None):
	if mu == None or sigma == None:
		mu = np.mean(X, axis=0)
		sigma = np.std(X, axis=0)
	return ((X - mu)/sigma, mu, sigma)

def gradient_check(theta, X, y, epsilon):
	return (cost_func(theta + epsilon, X, y) - cost_func(theta - epsilon, X, y))/float(2*epsilon)


def non_linear_data(X):
	return np.concatenate((X, X**2), axis=1)

X, y = get_data(2)

X = non_linear_data(X)

X_norm, mu, sigma = featureNormalize(X)

#change X to inculde bias (Add a column of ones to x)
X_one = np.concatenate((np.ones((X_norm.shape[0], 1)), X_norm), axis=1)

m, n = X.shape

epoch = 400
alpha = 0.75

#define theta intialize to random or zeros
#theta = np.zeros((n + 1, 2))
theta = np.random.random((n + 1, 2))

#overall/total average cost
cost = cost_func(theta, X_one, y)

print cost, '<- intial cost'

epsilon = 10**-4
#update theta
for _ in xrange(epoch):
	

	delta = 1.0/m * gradient(theta, X_one, y)
	theta += -alpha * delta 

	#cost from new theta
	#overall/total average cost
	cost = cost_func(theta, X_one, y)

	print cost


p1, _, _ = featureNormalize(non_linear_data(np.asarray([[0.051267, 0.69956]])), mu, sigma)
p2, _, _ = featureNormalize(non_linear_data(np.asarray([[0.18376, 0.93348]])), mu, sigma)

predict1 = func_h(theta, np.concatenate(([[1]], p1), axis=1))
predict2 = func_h(theta, np.concatenate(([[1]], p2), axis=1))

print predict1, predict2

points = X[(y[:, 0] == 1).ravel()]
plt.plot(points[:, 0], points[:, 1], 'yo')
points = X[(y[:, 1] == 1).ravel()]
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

p1, _, _ = featureNormalize(non_linear_data(np.asarray(temp)), mu, sigma)

p1 = func_h(theta, np.concatenate((np.ones((p1.shape[0], 1)), p1), axis=1))

temp = []
for t in p1:
	if t[0] >= t[1]:
		temp.append(0)
	else:
		temp.append(1)
p1 = np.asarray(temp)
p1 = p1.reshape(xx.shape)

plt.contourf(xx, yy, p1, cmap=plt.cm.Paired, alpha=0.8)
plt.show()