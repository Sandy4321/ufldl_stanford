import numpy as np 
import matplotlib.pyplot as plt

def get_data():
	X = []
	y = []
	fr = open('ex3Data/ex3x.dat')
	for x in fr:
		X.append(map(float, x.strip().split()))
	fr = open('ex3Data/ex3y.dat')
	for x in fr:
		y.append(map(float, x.strip().split()))
	return (np.asarray(X), np.asarray(y))

#define linear hypothysis return y
def linear_h(X, theta):
	temp = X.dot(theta)
	return temp.reshape((temp.shape[0], 1)) 

#mean square error function
def error_func(y_dash, y):
	return np.sum(0.5*(y_dash - y)**2, axis=1)


X, y = get_data()

#change X to inculde bias
X_one = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

#define theta intialize to random
theta = np.random.random((X_one.shape[1]))

#our intial prediction
y_dash = linear_h(X_one, theta) 

#intial error
error = error_func(y_dash, y)

#overall/total average cost
cost = np.mean(error, axis=0)

print cost

#cost generated
# plt.plot([x[0] for x  in X], [x[0] for x in y], 'r*')
# plt.show()