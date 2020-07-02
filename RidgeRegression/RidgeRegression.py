import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

def mean_absolute_percentage_error(y_true, y_pred):

	"""	MAPE = (1/n) * ((|y_true - y_pred|)/y_true) """

	y_true, y_pred = np.array(y_true), np.array(y_pred)
	ape =  np.abs((y_true - y_pred)/y_true)
	mape = np.mean(ape) * 100

	return mape

def real_estate_pre_processing(df):

	""" partioning data into features and target """

	X = df.drop([df.columns[0], df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class Ridge:

	def __init__(self, learning_rate = 1e-3, alpha = 1.0, max_iter = 1000):

		self.num_feats = int
		self.train_size = int
		self.weights = np.array 
		self.y_train = np.array 
		self.input_matrix = np.array

		self.learning_rate = learning_rate   #Learning rate for gradient descent
		self.alpha = alpha 	 #Regularization parameter, to control bias-variance tradeoff
		self.max_iter = max_iter 	#Number of iterations to run gradient descent
		self.cost_threshold = 0.1 * learning_rate  #stopping criterion for gradien descent

	def fit(self, X, y):

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
				l(w) = (1/n) * (((y - wX)^2) + alpha * (w^2))

			Gradient:
				delta_w = dl/dw = (2/n)*( ((y - wX)*(-X)) + alpha * w) 
							
							 (or)

				delta_w = dl/dw = (2/n)*( ((wX - y)*(X)) + alpha * w)

			Gradient Descent:
				w = w - (learning_rate * delta_w)

		"""

		y_pred = (self.weights * self.input_matrix).sum(axis = 1)  # y_pred = wX

		cost = (1/self.train_size) * (((self.y_train - y_pred) ** 2).sum(axis = 0) + (self.alpha * (self.weights ** 2)).sum(axis = 0))  

		err = (y_pred - self.y_train).reshape(-1, 1)  # err = wX - y

		delta_w = (2/self.train_size) * (((err * self.input_matrix).sum(axis = 0)) + (self.alpha * self.weights)) #delta_w = (2/n)*( ((wX - y)*(X)) + alpha * w)

		self.weights = self.weights - (self.learning_rate * delta_w) 

		return cost


	def predict(self, X):

		""" Make predictions on given X using trained model """

		size = X.shape[0]
		X = np.append(X, np.ones(size).reshape(-1, 1), axis = 1)

		y_pred = (self.weights * X).sum(axis = 1)

		return y_pred 
		



if __name__ == '__main__':

	#Synthetic DataSet For Regression 
	print("\nSynthetic Dataset")
	data = pd.read_table("../Data/regression_data.txt")
	#print(data)

	#Separate Features and Target
	X = data.drop('Y', axis = 1)
	y = data['Y']


	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


	# Create a Ridge Regression Model Object
	ridge_reg = Ridge(learning_rate = 1e-4, alpha = 10.0)

	#Train our Ridge Regression Model
	ridge_reg.fit(X_train, y_train)


	print("Mean Absolute Percentage Error(for test data): {}".format(mean_absolute_percentage_error(y_test, ridge_reg.predict(X_test))))

	print('Ridge Regression Model Coefficients (W): {}'.format(ridge_reg.weights[:-1]))
	print('Ridge Regression Model Intercept (b): {}'.format(ridge_reg.weights[-1]))

	
	#Query Sample 1: [80]
	y_pred = ridge_reg.predict(np.array([[80]]))
	print("predicted target is {}".format(y_pred))


	#################################################################################################################

    #Application to Real World Dataset
	print("\nReal Estate Dataset:")
	df = pd.read_csv('../Data/datasets_Real_Estate.csv')

	#Pre-processing
	X, y = real_estate_pre_processing(df)


	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


	ridge_reg = Ridge(learning_rate = 1e-7, alpha = 100.0)
	ridge_reg.fit(X_train, y_train)

	y_pred = ridge_reg.predict(X_test)



	print("\nMean Absolute Percentage Error(for train data): {}".format(mean_absolute_percentage_error(y_train, ridge_reg.predict(X_train))))
	print("Mean Absolute Percentage Error(for test data): {}".format(mean_absolute_percentage_error(y_test, ridge_reg.predict(X_test))))

	print('Ridge Regression Model Coefficients (W): {}'.format(ridge_reg.weights[:-1]))
	print('Ridge Regression Model Intercept (b): {}'.format(ridge_reg.weights[-1]))	



	#If you apply normalization to your trained data, then you need to apply same normalization to each query as well.
	#Query Sample 1:
	y_pred = ridge_reg.predict(np.array([[2012.667,5.6,90.45606,9,24.97433,121.5431]]))
	print("Predicted target is: {}".format(y_pred))