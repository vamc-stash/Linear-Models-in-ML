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

def min_max_scaler(X):

	""" Min-Max Normalization"""
	for column in X.columns:
		X[column] = (X[column] - np.min(X[column])) / (np.max(X[column]) - np.min(X[column]))

	return X




class LinearRegression:

	def __init__(self):
		self.coef_ = []
		self.intercept_ = float

	def fit(self, X, y):

		"""
			Calculates Coefficients(w) and intercept(b) of linear model

			y = wX + b

			wi = (sum[j=0 to n]((xj - x_meani)*(y - y_mean)) / (sum[j=0 to n]((xj - x_meani) ** 2)
			b  =  y - wX

			where x_meani = mean of ith column features
				  y_mean = mean of targets
				  xj = feature value of jth row and ith column
				  y = target value of jth row
				  wi = coefficient of ith feature
				  b = intercept
				  n = number of samples
		"""

		X = X.T.values.tolist()
		y = y.values.tolist()

		X_mean = np.mean(X, axis = 1).tolist()
		y_mean = np.mean(y, axis = 0)

		for x, x_mean in zip(X, X_mean):

			covariance  = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])
			variance = sum([(x[i] - x_mean) ** 2  for i in range(len(x))])

			w = covariance/variance
			
			self.coef_.append(w) 

		self.intercept_ = y_mean - sum([w * x_mean for w, x_mean in zip(self.coef_, X_mean)])

	def predict(self, X):

		""" predicts target values for given input X"""

		predictions = []

		for Xi in X:
			y_pred = self.intercept_

			for w, x in zip(Xi, self.coef_):
				y_pred += w*x
			predictions.append(y_pred)

		return predictions


	def score(self, X, y):

		""" calcuates Root Mean Square Error between actual and predicted targets 

			RMSE = sqrt(mean( y_true - y_predict) ** 2))

		"""

		y_preds = self.predict(X.values.tolist())
		sum_error = 0

		for y_pred, y_true in zip(y_preds, y.values.tolist()):
			sum_error += ((y_pred - y_true) ** 2)

		mean_error = sum_error/len(y)
		root_mean_square_error = math.sqrt(mean_error)

		return root_mean_square_error


if __name__ == '__main__':


	#Simple Linear Regression

	#Synthetic DataSet For Regression 
	print("\nSynthetic Dataset")
	data = pd.read_table("../Data/regression_data.txt")
	#print(data)

	#Data Visualization
	plt.scatter(data['X'], data['Y'], c = 'orange', marker = 'o', label = 'Data Points')
	plt.gca().set_xlabel('X')
	plt.gca().set_ylabel('Y')
	plt.title('Data for linear regression')
	plt.legend()
	plt.savefig('DataVisualization.png')


	#Separate Features and Target
	X = data.drop('Y', axis = 1)
	y = data['Y']

	# XX = poly_feats(X, degree = 3)
	# print(XX)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


	# Create a LinearRegression Model Object
	lin_reg = LinearRegression()

	#Train our LinearRegression Model
	lin_reg.fit(X_train, y_train)


	#print("Mean Absolute Percentage Error(for test data): {}".format(mean_absolute_percentage_error(y_test, lin_reg.predict(X_test))))

	print('Linear Regression Model Coefficients (W): {}'.format(lin_reg.coef_))
	print('Linear Regression Model Intercept (b): {}'.format(lin_reg.intercept_))



	#Model Visualization (with One feature Vs target)
	plt.plot(X, lin_reg.coef_ * X + lin_reg.intercept_, 'r-', label = 'Linear Regression Model')
	plt.gca().set_xlabel('X')
	plt.gca().set_ylabel('Y')
	plt.title('Linear Regression model')
	plt.legend()
	plt.savefig('LinearRegression.png')

	#Evaluating Model through RMSE
	Training_error = lin_reg.score(X_train, y_train)
	print("RMSE for training data: {:.3f}".format(Training_error))

	Test_error = lin_reg.score(X_test, y_test)
	print("RMSE for training data: {:.3f}".format(Test_error))


	#Query Sample 1: [80]
	y_pred = lin_reg.predict([[80]])
	print("predicted target is {}".format(y_pred))

    #################################################################################################################

    #Multi-Linear Regression

    #Application to Real World Dataset
	print("\nReal Estate Dataset:")
	df = pd.read_csv('../Data/datasets_Real_Estate.csv')

	#print(df.head())

	#Pre-processing
	X, y = real_estate_pre_processing(df)

	#Normalization
	#X = min_max_scaler(X)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



	lin_reg = LinearRegression()
	lin_reg.fit(X_train, y_train)

	print('Linear Regression Model Coefficients (W): {}'.format(lin_reg.coef_))
	print('Linear Regression Model Intercept (b): {}'.format(lin_reg.intercept_))

	#Evaluating Model through RMSE
	Training_error = lin_reg.score(X_train, y_train)
	print("RMSE for training data: {:.3f}".format(Training_error))

	Test_error = lin_reg.score(X_test, y_test)
	print("RMSE for training data: {:.3f}".format(Test_error))
	

	#If you apply normalization to your trained data, then you need to apply same normalization to each query as well.
	#Query Sample 1:
	y_pred = lin_reg.predict([[2012.667,5.6,90.45606,9,24.97433,121.5431]])
	print("Predicted target is: {}".format(y_pred))