'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.theta = None
        self.regLambda = reg_lambda
        self.degree = degree
        self.mean = None
        self.std = None

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        outputX = X[:]
        for i in range(2, degree + 1):
            outputX = np.hstack((outputX,X**i))
        return outputX


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """

        X = self.polyfeatures(X, self.degree)
        # standardization
        print("X: ",X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        print("Mean: ", self.mean, "std: ", self.std)
        X = (X - self.mean) / self.std
        print("X: ",X)
        n = len(X)
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]
        n, d = X_.shape
        print("X_ ",X_, " y: ", y, " n: ", n," d: ", d, " lambda: ", self.regLambda)
        reg_matrix = self.regLambda * np.identity(d )
        reg_matrix[0, 0] = 0
        self.theta = np.linalg.pinv((X_.T @ X_) + reg_matrix) @ (X_.T @ y)
        print("Theta: ",self.theta)


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """

        n = len(X)
        X = self.polyfeatures(X, self.degree)

        X = (X - self.mean) / self.std
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        # predict
        return X_ @ self.theta


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------
def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(3, n+1):
        print("i = ", i, " Degree= ", degree, " lambda: ", reg_lambda)
        model = PolynomialRegression(degree, reg_lambda)
        model.fit(Xtrain[:i], Ytrain[:i])

        trainPredicted = model.predict(Xtrain[:i])
        singleErrorFromTrain = np.mean((trainPredicted- Ytrain[:i])**2)
        errorTrain[i-1] = singleErrorFromTrain

        test_predicted = model.predict(Xtest[:i])
        singleErrorFromTest = np.mean((test_predicted - Ytest[:i])**2)
        errorTest[i-1] = singleErrorFromTest

    return errorTrain, errorTest


