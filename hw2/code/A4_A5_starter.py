import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_x(n,d):
    return np.random.standard_normal((n,d))

def generate_y(n, d, k, X):
    w = np.zeros((d,1))
    for i in range(0, d):
        if (0 <= i and i < k):  # from first to kth w
            w[i] = (i + 1)/k
        else:
            w[i] = 0
    # create y_i
    y = np.zeros((n, 1))
    for i in range(n):
        y[i] = w.T @ X[i] + np.random.standard_normal()
    return y, w


def compute_initial_lamb(x, y):
    n, d = x.shape
    lamb_array = np.zeros((d, 1))
    for k in range(d):
        lamb_temp = 2 * abs(x[:, k].T @ (y - np.mean(y)))
        lamb_array[k] = lamb_temp
    return max(lamb_array)

class LASSO:

    def __init__(self, lamb=1E-8, delta=0.005):
        self.lamb = lamb
        self.last_w = None
        self.b = 0.0
        self.delta = delta
        self.loss_list = []
        self.last_selected_coef = []
        self.selected_feature_index = [1, 3, 5, 7, 12]

    def coordinate_descent(self, X, y, initial_w):
        n, d = X.shape
        # W = np.zeros((d, 1))
        W = initial_w
        A = 2 * np.sum(np.power(X, 2), axis=0)  # Power and then sum up the rows # C = np.zeros((d, 1))
        # Iterate until convergence
        loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(abs(W))

        converged = False
        while not converged:
            self.b = np.average(y - X.dot(W))
            # converged = True
            loss_prev = loss
            # Compute C, w_new
            prev_w = np.copy(W)  # Diff conv
            for k in range(d):
                sliced_x = X[:, k]
                prev_wk = np.copy(W[k])
                W[k] = 0
                c_k = np.dot((y - (self.b * np.ones((n, 1)) + X.dot(W))).T, sliced_x)
                if 2 * c_k + self.lamb < 0:
                    W[k] = (c_k * 2 + self.lamb) / A[k]
                elif 2 * c_k - self.lamb > 0:
                    W[k] = (c_k * 2 - self.lamb) / A[k]
                else:
                    W[k] = 0

            if sum(abs(W - prev_w)) <= sum(abs(self.delta * prev_w)):
                converged = True
                print(W)

            loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(abs(W))
            self.loss_list.append(loss)
            # print(loss)
            self.last_w = W
            self.last_selected_coef = self.last_w.T[0][self.selected_feature_index]

    def predict(self, X):
        return X.dot(self.last_w)

    def num_of_nonzeros(self):
        return np.count_nonzero(self.last_w)



