import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random

def load_data(size=1):
    print("Loading Date!")
    mndata = MNIST('python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    if (size != 1):
        return X_train[:int(len(X_train)*size)], \
               labels_train[:int(len(labels_train)*size)], \
               X_test[:int(len(X_test)*size)], \
               labels_test[:int(len(labels_test)*size)]
    else:
        return X_train, labels_train, X_test, labels_test
X_train, y_train, X_test, y_test = load_data(size=1)
print("Load Data Compelete!")

# A6.a
n, d = X_train.shape
mu = np.zeros(d)
for i in range(d):
    mu[i] = np.mean(X_train[:, i])

demean_X_train = X_train[:]
for row in range(X_train.shape[0]):
    demean_X_train[row] = X_train[row] - mu
demean_X_test = X_test[:]
for row in range(X_test.shape[0]):
    demean_X_test[row] = X_test[row] - mu

sigma = (demean_X_train.T @ demean_X_train) / 60000


U, S, V = np.linalg.svd(demean_X_train / np.sqrt(60000), False)
eigenvalues = S**2
print("Lambda 1: ", eigenvalues[0])
print("Lambda 2: ", eigenvalues[1])
print("Lambda 10: ", eigenvalues[9])
print("Lambda 30: ", eigenvalues[29])
print("Lambda 50: ", eigenvalues[49])
print("Sum of Lambdas: ", sum(eigenvalues))

# Lambda 1:  5.116787728342091
# Lambda 2:  3.7413284788648014
# Lambda 10:  1.24272937641733
# Lambda 30:  0.36425572027888947
# Lambda 50:  0.16970842700672756
# Sum of Lambdas:  52.72503549512679
# A6.b


# ========================= A6c =========================
# plot mse
mean_squared_error_list_train = []
mean_squared_error_list_test = []
for k in range(0, 100):
    print("K: ", k)
    reconstructed = np.dot(V[:k, :].T.dot(V[:k, :]), demean_X_train[:, :].T).T
    MSE_train = np.sum((reconstructed - demean_X_train) ** 2) / demean_X_train.shape[0]
    mean_squared_error_list_train.append(MSE_train)

    reconstructed_test = np.dot(V[:k, :].T.dot(V[:k, :]), demean_X_test[:, :].T).T
    MSE_test = np.sum((reconstructed_test - demean_X_test) ** 2) / demean_X_test.shape[0]
    mean_squared_error_list_test.append(MSE_test)

plt.plot(range(1, 101), mean_squared_error_list_train, label="Train MSE")
plt.plot(range(1, 101), mean_squared_error_list_test, label="Test MSE")
plt.xlabel("Number of top lambdas")
plt.ylabel("MSE")
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A6c_1.png")
plt.show()


# plot construction error
eigenvalues_sum = sum(eigenvalues)
fraction_error_list = []
for k in range(0,100):
    fraction_error = 1 - np.sum(eigenvalues[:(k+1)]) / eigenvalues_sum
    fraction_error_list.append(fraction_error)

plt.plot(range(1, 101), fraction_error_list)
plt.xlabel("Number of Lambdas")
plt.ylabel("fraction_error")
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A6c_2.png")
plt.show()

# ========================= A6d =========================
for k in range(10):
    plt.imshow(V[k,:].reshape((28,28)))
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A6d/A6d_"+str(k)+".png")
    plt.show()


# ========================= A6e =========================
y_train[5] # =2
y_train[13] # =6
y_train[15] # =7

for y in [5, 13, 15]:  # this is the index of y in the X_train
    plt.imshow(X_train[y,:].reshape((28,28)))
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A6e/A6e_"+str(y)+".png")
    plt.show()

    for k in [5, 15, 40, 100]:
        plt.imshow(np.dot(V[:k, :].T.dot(V[:k, :]), X_train[:, :].T).T[y].reshape((28, 28)))
        # plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A6e/A6e_" + str(y) + "_"+str(k)+".png")
        plt.show()

plt.imshow(np.dot(V[:k, :].T.dot(V[:k, :]), X_train[:, :].T).T[y].reshape((28, 28)))
plt.show()



# k = 100
plt.imshow(np.dot(V[:k, :].T.dot(V[:k, :]), X_train[:, :].T).T[0].reshape((28, 28)))
i = 12
plt.imshow((demean_X_train[i] + mu).reshape((28, 28)))
plt.show()
#
# np.dot(V[:k, :].T.dot(V[:k, :]), X_train[:, :].T).T[5]
np.sum(demean_X_train[1] - demean_X_train[2])

