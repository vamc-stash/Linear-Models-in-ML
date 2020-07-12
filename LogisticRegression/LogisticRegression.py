import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

eps = np.finfo(float).eps

def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop([df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class Logistic:

	"""
		Logistic Regression is a Classification ML model.
	"""

	def __init__(self, learning_rate = 1e-3, max_iter = 2000):

		self.num_feats = int
		self.train_size = int
		self.weights = np.array 
		self.y_train = np.array 
		self.input_matrix = np.array

		self.learning_rate = learning_rate   #Learning rate for gradient descent
		self.max_iter = max_iter 	#Number of iterations to run gradient descent
		self.cost_threshold = 0.1 * learning_rate  #stopping criterion for gradien descent

	def sigmoid(self, x):

		"""
			Logistic function for binary classification.

			Sigmoid = 1/(1 + e^(-x))  -> It outputs values between 0 and 1

		"""

		return 1 / (1 + np.exp(-x))

	def fit(self, X, y, logistic_function = "sigmoid"):

		"""
			Adjust weights to training data

		"""

		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]
		self.input_matrix = np.append(X, np.ones(self.train_size).reshape(-1, 1), axis = 1)   #Add Column with Ones for intercept term 
		self.y_train = y.to_numpy()
		self.weights = np.zeros(self.num_feats + 1) #Extra +1 for the intercept


		#optimize weights
		prev_cost = float("inf")
		for i in range(self.max_iter):
			cost = self._update_weights()

			if i%100 ==0 or i == self.max_iter:
				print("Cost after {} iterations is: {}".format(i, cost))
			if abs(prev_cost -cost) < self.cost_threshold*prev_cost:
				print("Cost after {} iterations is: {}".format(i, cost))
				break
			prev_cost = cost


	def _update_weights(self):

		"""
			Cost Function:
				yhat = sigmoid(wX)
				l(w) = -(1/n) * (y*log(yhat) + (1-y)*log(1-yhat))

			Gradient:
				delta_w = dl/dw = (1/n)*((yhat - y) * X)) 
		
			Gradient Descent:
				w = w + (learning_rate * delta_w)

		"""

		y_pred = self.sigmoid((self.weights * self.input_matrix).sum(axis = 1))  # y_pred = sigmoid(wX)

		cost = -(1/self.train_size) * (self.y_train*np.log(y_pred+eps) + (1-self.y_train)*np.log(1-y_pred+eps)).sum(axis = 0)

		delta_w = (1/self.train_size) * (((y_pred - self.y_train).reshape(-1, 1) * self.input_matrix).sum(axis = 0))  #delta_w = (1/n)*((yhat - y) * X)) 

		self.weights = self.weights - (self.learning_rate * delta_w) 

		return cost

	def _update_weightss(self):

		"""
			******** This Method is wrote only for understanding and it is not utilized anywhere in the algorithm ***********

			Cost Function:
				yhat = sigmoid(wX)
				l(w) = (1/n) * (y - yhat)^2)

			Gradient:
				delta_w = dl/dw = (2/n)*((y - yhat) * yhat * (1-yhat) * X)) 
		
			Gradient Descent:
				w = w + (learning_rate * delta_w)

			If we try to use the cost function of the linear regression in ‘Logistic Regression’ then it would 
			be of no use as it would end up being a non-convex function with many local minimums, in which it 
			would be very difficult to minimize the cost value and find the global minimum.

		"""

		y_pred = self.sigmoid((self.weights * self.input_matrix).sum(axis = 1))  # y_pred = sigmoid(wX)

		cost = (1/self.train_size) * ((self.y_train - y_pred) ** 2).sum(axis = 0)

		temp = ((self.y_train - y_pred) * y_pred * (1-y_pred)).reshape(-1, 1)  # temp = (y - ypred) * ypred * (1-ypred)

		delta_w = (2/self.train_size) * ((temp * self.input_matrix).sum(axis = 0))  #delta_w = (2/n)*((y - yhat) * yhat * (1-yhat) * X))

		self.weights = self.weights + (self.learning_rate * delta_w) 

		return cost


	def predict(self, X):

		""" Make predictions on given X using trained model """

		size = X.shape[0]
		X = np.append(X, np.ones(size).reshape(-1, 1), axis = 1)

		y_pred = self.sigmoid((self.weights * X).sum(axis = 1))

		y_pred[np.where(y_pred >= 0.5)] = 1.0
		y_pred[np.where(y_pred < 0.5)] = 0.0

		return y_pred 
		

if __name__ == '__main__':

	#Synthetic DataSet For Binary Classification 
	print("\nSynthetic Dataset:")
	"""
		Input: Two Features
		Target: 1 or O

	"""
	data = pd.read_table("../Data/classification_data.txt")

	#Separate Features and Target
	X, y = pre_processing(data)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


	# Create a Logistic Regression Model Object
	logistic_clf = Logistic(learning_rate = 1e-2, max_iter = 2000)

	#Train our Logistic Regression Model
	logistic_clf.fit(X_train, y_train)


	print("Train Accuracy: {}%".format(accuracy_score(y_train, logistic_clf.predict(X_train))))
	print("Test Accuracy: {}%".format(accuracy_score(y_test, logistic_clf.predict(X_test))))

	print('Logistic Regression Model Coefficients (W): {}'.format(logistic_clf.weights[:-1]))
	print('Logistic Regression Model Intercept (b): {}'.format(logistic_clf.weights[-1]))

	
	#Query Sample 1: [6.3223, 4.32]
	y_pred = logistic_clf.predict(np.array([[6.3223, 4.32]]))
	print("predicted target is {}".format(y_pred))

	################################################################################################################

	#Real World Dataset
	print("\nCancer Dataset:")
	df = pd.read_csv("../Data/datasets_cancer.csv")

	#pre-processing
	df.reset_index(inplace = True)
	lookup_map = {'M': 1.0, 'B': 0.0}

	df['diagnosis'] = df['diagnosis'].map(lookup_map)
	X = df.drop([df.columns[1], df.columns[2], df.columns[-1]], axis = 1)
	y = df[df.columns[2]]
	
	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


	# Create a Logistic Regression Model Object
	logistic_clf = Logistic(learning_rate = 1e-5, max_iter = 5000)

	#Train our Logistic Regression Model
	logistic_clf.fit(X_train, y_train)


	print("Train Accuracy: {}%".format(accuracy_score(y_train, logistic_clf.predict(X_train))))
	print("Test Accuracy: {}%".format(accuracy_score(y_test, logistic_clf.predict(X_test))))

	print('Logistic Regression Model Coefficients (W): {}'.format(logistic_clf.weights[:-1]))
	print('Logistic Regression Model Intercept (b): {}'.format(logistic_clf.weights[-1]))