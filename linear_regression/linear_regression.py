import numpy as np 
import matplotlib.pyplot as plt

# def get_data():
# 	X = []
# 	y = []
# 	fr = open('ex3Data/ex3x.dat')
# 	for x in fr:
# 		X.append(map(float, x.strip().split()))
# 	fr = open('ex3Data/ex3y.dat')
# 	for x in fr:
# 		y.append(map(float, x.strip().split()))
# 	return (np.asarray(X), np.asarray(y))

def get_data():
	X = []
	y = []
	fr = open('ex3Data/ex1data1.txt')
	for x in fr:
		data = map(float, x.strip().split(','))
		X.append(data[0])
		y.append(data[1])
	X = np.asarray(X)
	X = X.reshape((X.shape[0], 1))
	y = np.asarray(y)
	y = y.reshape((y.shape[0], 1))
	return (X, y)

#define linear hypothysis return y
def linear_h(X, theta):
	return X.dot(theta)

#mean square error function
def error_func(y_dash, y):
	return np.sum(0.5*(y_dash - y)**2, axis=1)

def gradient(theta, X, y):
	return X.T.dot(linear_h(X, theta) - y)

X, y = get_data()

#change X to inculde bias (Add a column of ones to x)
X_one = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


epoch = 1500
alpha = 0.01
m, n = X_one.shape

#define theta intialize to random
# theta = np.random.random((n, 1))
theta = np.zeros((n, 1))

#our intial prediction
y_dash = linear_h(X_one, theta) 

#intial error
error = error_func(y_dash, y)

#overall/total average cost
cost = np.mean(error, axis=0)

print cost, '<- intial cost'

#update theta

for _ in xrange(epoch):
	
	delta = - alpha/m * gradient(theta, X_one, y)
	theta += delta 

	#intial error
	error = error_func(linear_h(X_one, theta), y)

	#overall/total average cost
	cost = np.mean(error, axis=0)

	# print cost

predict1 = np.asarray([1, 3.5]).dot(theta)
predict2 = np.asarray([1, 7]).dot(theta)

print predict1, predict2

size = 25
plt.figure(figsize=(10, 8))
plt.plot(y, X, 'r*')
X_temp = np.asarray(xrange(0, size)).reshape((size, 1))
y_temp = np.concatenate((np.ones((size, 1)), X_temp), axis=1).dot(theta)
plt.plot(y_temp, X_temp, label='predicted')
plt.legend(loc='upper left')
plt.ylabel('House price ($1000s)')
plt.xlabel('House #')
plt.show()