import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def get_data():
	data_original = np.loadtxt('housing.data')
	data = np.insert(data_original, 0, 1, axis=1)
	np.random.shuffle(data)

	train_X = data[:400, :-1]
	train_y = data[:400, -1]

	test_X = data[400:, :-1]
	test_y = data[400:, -1]
	return (scale(train_X), scale(train_y.reshape(train_y.shape[0], 1)))

#define linear hypothysis return y
def linear_h(X, theta):
	return X.dot(theta)

#mean square error function
def error_func(y_dash, y):
	return np.sum(0.5*(y_dash - y)**2, axis=1)

def gradient(theta, X, y):
	return X.T.dot(linear_h(X, theta) - y)

X, y = get_data()

alpha = 0.05
m, n = X.shape


#define theta intialize to random
theta = np.random.random((n, 1))

#our intial prediction
y_dash = linear_h(X, theta) 

#intial error
error = error_func(y_dash, y)

#overall/total average cost
cost = np.mean(error, axis=0)

print cost

prev_cost = None
#update theta
while(True):

	delta = - alpha/m * gradient(theta, X, y)
	theta += delta 

	#intial error
	error = error_func(linear_h(X, theta), y)

	#overall/total average cost
	cost = np.mean(error, axis=0)

	print cost
	if prev_cost != None and prev_cost - cost <= 0.000000005:
		break
	prev_cost = cost

print theta


plt.figure(figsize=(10, 8))
plt.scatter(np.arange(y.size), sorted(y), c='b', edgecolor='None', alpha=0.5, label='actual')
plt.scatter(np.arange(y.size), sorted(X.dot(theta)), c='g', edgecolor='None', alpha=0.5, label='predicted')
plt.legend(loc='upper left')
plt.ylabel('House price ($1000s)')
plt.xlabel('House #')
plt.show()