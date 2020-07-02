import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

from itertools import combinations_with_replacement

import sys
sys.path.append('../')

from RidgeRegression.RidgeRegression import Ridge
from LassoRegression.LassoRegression import Lasso


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

	



class PolynomialFeatures:

	def __init__(self, degree = 2):

		"""
			Args: degree = 2(default)
		"""

		self.degree = degree
		self.num_input_feats = int 
		self.num_output_feats = int

	def fit_transform(self, X):

		"""
			Generates new feature matrix consisiting of all combinations of
			the features with degree less than or equal to specified degree(or 2 by default) 

			Ex. input sample = [a, b]
			    output sample(with degree 3) = [1, a, b, a^2, ab, b^2, a^3, a^2b, ab^2, b^3]


			Args: 
				X (np.array) - matrix with features to be transformed

			Returns:
				X_new (np.array) - matrix with transformed features

		"""

		sample_size, self.num_feats = np.shape(X)

		combs_obj = [combinations_with_replacement(range(np.shape(X)[1]), i) for i in range(0, self.degree + 1)]
		combinations = [item  for combination in combs_obj for item in combination]

		self.num_output_feats = len(combinations)

		X_new = np.empty((sample_size, self.num_output_feats))

		#Note: np.prod([]) = 1.0 (The product of an empty array is the neutral element 1.0)
		for i, index_combs in enumerate(combinations):
			index_combs = list(index_combs)
			X_new[:, i] = np.prod(X.iloc[:, index_combs], axis=1)

		return pd.DataFrame(X_new)


if __name__ == "__main__":

	print("\nReal Estate Dataset:")
	df = pd.read_csv('../Data/datasets_Real_Estate.csv')

	#Pre-processing
	X, y = real_estate_pre_processing(df)

	#Create a PolynomialFeatures Object
	poly_obj = PolynomialFeatures()
	X_new = poly_obj.fit_transform(X)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0)


	print("\nRidgeRegression with PolynomialFeatures:-")
	ridge_reg = Ridge(learning_rate = 1e-14, alpha = 100.0)
	ridge_reg.fit(X_train, y_train)

	print("Mean Absolute Percentage Error(for train data): {}".format(mean_absolute_percentage_error(y_train, ridge_reg.predict(X_train))))
	print("Mean Absolute Percentage Error(for test data): {}".format(mean_absolute_percentage_error(y_test, ridge_reg.predict(X_test))))

	print('Ridge Regression Model Coefficients (W): {}'.format(ridge_reg.weights[:-1]))
	print('Ridge Regression Model Intercept (b): {}'.format(ridge_reg.weights[-1]))

	#Query Sample 1:
	query1 = np.array([[2012.667,5.6,90.45606,9,24.97433,121.5431]])
	poly_quer1 = poly_obj.fit_transform(pd.DataFrame(query1))
	y_pred = ridge_reg.predict(poly_quer1)
	print("Predicted target is: {}".format(y_pred))




	print("\nLassoRegression with PolynomialFeatures:-")
	lasso_reg = Lasso(learning_rate = 1e-14, alpha = 100.0)
	lasso_reg.fit(X_train, y_train)

	print("Mean Absolute Percentage Error(for train data): {}".format(mean_absolute_percentage_error(y_train, lasso_reg.predict(X_train))))
	print("Mean Absolute Percentage Error(for test data): {}".format(mean_absolute_percentage_error(y_test, lasso_reg.predict(X_test))))

	print('Lasso Regression Model Coefficients (W): {}'.format(lasso_reg.weights[:-1]))
	print('Lasso Regression Model Intercept (b): {}'.format(lasso_reg.weights[-1]))

	#Query Sample 1:
	query1 = np.array([[2012.667,5.6,90.45606,9,24.97433,121.5431]])
	poly_quer1 = poly_obj.fit_transform(pd.DataFrame(query1))
	y_pred = lasso_reg.predict(poly_quer1)
	print("Predicted target is: {}".format(y_pred))

