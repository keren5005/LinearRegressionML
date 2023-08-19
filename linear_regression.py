from typing import Union
import numpy as np

"""
The class takes in parameters for learning rate, number of epochs, and threshold, and uses a closed-form solution 
or gradient descent method to fit the model depending on the invertibility of the design matrix.
"""
class LinearRegression:
    #the constructor of the class, responsible for initializing the class attributes.
    def __init__(self, learning_rate: float, n_epochs: int, threshold: Union[None, float] = None):
        self.weights_: Union[None, np.array] = None
        self.__alpha = learning_rate
        self.__epochs = n_epochs
        self.__eta = -np.inf if threshold is None else threshold

    def fit(self, X: np.array, y: np.array, always_gd: bool = False, verbose: bool = False):
        #This fit method is responsible for fitting the model on the input data.
        # X is the input feature matrix, and y is the target variable vector.
        n = X.shape[0]  # number of samples
        m = X.shape[1]  # number of features
        if y.shape[0] != n:
            raise Exception("X and y should have same number of rows")

        x = np.c_[np.ones(n), X]
        xt = np.transpose(x)
        xtx = np.matmul(xt, x)

        if np.linalg.det(xtx) != 0 and not always_gd:
            inv = np.linalg.inv(xtx)
            self.weights_ = np.matmul(np.matmul(inv, xt), y)
        else:
            e = 0
            delta = np.inf
            self.weights_ = np.zeros(m + 1)
            while e < self.__epochs and delta > self.__eta:
                grad = np.dot(self.weights_.T, xtx) - np.dot(y.T, x)
                self.weights_ = self.weights_ - self.__alpha * grad
                e = e + 1
                delta = self.__error_internal(x, y)
                if verbose:
                    if e % 1000 == 0:
                        print(f'epoch={e}, error={delta}')
            self.weights_ = np.squeeze(self.weights_)

    def __error_internal(self, x, y):
        #used by the fit method to calculate the error of the model.
        z = np.dot(x, self.weights_) - y
        return np.dot(z, z)

    def error(self, y, yhat):
        #This method calculates the mean squared error between the predicted yhat and the actual y.
        z = yhat - y
        return np.dot(z, z)

    def predict(self, X: np.array) -> np.array:
        #This method is responsible for predicting the output of the model given the input feature matrix X.
        n = X.shape[0]
        x = np.c_[np.ones(n), X]
        return np.dot(x, self.weights_)

    def score(self, y, y_hat):
        #This method calculates the R-squared score of the model,
        # #given the predicted y_hat and the actual y. It is used to evaluate the performance of the model.
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        ybar = np.mean(y)
        ssreg = np.sum(np.power(y - y_hat, 2))
        sstot = np.sum(np.power(y - ybar, 2))
        return 1.0 - ssreg / sstot
