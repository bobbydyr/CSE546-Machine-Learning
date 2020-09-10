import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class RegularizedLogisticRegression:

    def __init__(self, step_size = 0.01, stopping = 0.05):
        self.step_size = step_size
        self.stopping = stopping
        self.lamb = 0.1

    def GD(self, X, y, iter):
        n, d = X.shape
        old_b = 0
        new_b = 0
        old_w = np.zeros((d, 1))
        new_w = np.zeros((d, 1))
        not_converge = True
        iteration = 0
        while iteration < iter:
            iteration += 1
            print("============== >>>   iteration: ", iteration)
            exp = np.exp(-y * (X @ new_w + new_b) )
            u = 1 / ( 1 + exp)
            gradient_w = (1 / (X.shape[0])) * np.dot( ((1 - u)*(-y)).T, X) + 2 * self.lamb * new_w
            gradient_w = np.mean(gradient_w, axis=0).reshape((d,1))
            # gradient_w = np.mean( (1-u)*(-y) * X, axis=0).reshape((d, 1)) + 2*self.lamb*new_w
            gradient_b = np.mean((1-u) * (-y))

            learned_w = self.step_size * gradient_w
            new_w = new_w - learned_w
            learned_b = self.step_size * gradient_b
            new_b  = new_b - learned_b

            old_w = new_w.copy()
            old_b = new_b.copy()

            loss = np.mean(np.log(1 + np.exp(-y * (new_b + np.dot(X, new_w))))) + self.lamb * np.linalg.norm(new_w, 2)
            print("Objective Function: ", loss)

            print("W change norm: ", np.linalg.norm(old_w - new_b, 2))
            if np.linalg.norm(old_w - new_b, 2) < self.stopping:
                not_converge = False

        return new_w, new_b

    def get_gradient(self, x, y, w, i):
        u_i = 1 + np.exp( -y[i] * (b + x[i,:] @ w ))


if __name__ == '__main__':
    import mnist
    import numpy as np

    mndata = mnist.MNIST("./python-mnist/data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0

    X_train = np.vstack( (X_train[labels_train == 2], X_train[labels_train == 7])).astype("float32")
    y_train = np.hstack((labels_train[labels_train == 2], labels_train[labels_train == 7])).astype("float32")
    y_train[y_train == 2] = -1
    y_train[y_train == 7] = 1
    y_train = y_train.reshape((len(y_train), 1))

    X_test = np.vstack((X_test[labels_test == 2], X_test[labels_test == 7])).astype("float32")
    y_test = np.hstack((labels_test[labels_test == 2], labels_test[labels_test == 7])).astype("float32")
    y_test[y_test == 2] = -1
    y_test[y_test == 7] = 1
    y_test = y_test.reshape((len(y_test), 1))

    classification_rate_train_list = []
    classification_rate_test_list = []
    object_train_list = []
    object_test_list = []

    iteration_list = [i for i in range(1, 151)]
    iteration_list = iteration_list[-2:]

    for i in iteration_list:
        model = RegularizedLogisticRegression(step_size = 0.05, stopping = 0.05)
        best_coef = model.GD(X_train, y_train, iter = i)

        obj_train = np.mean(np.log(1 + np.exp(-y_train * (best_coef[1] + np.dot(X_train, best_coef[0]))))) + 0.1 * np.linalg.norm(best_coef[0], 2)
        obj_test = np.mean(np.log(1 + np.exp(-y_test * (best_coef[1] + np.dot(X_test, best_coef[0]))))) + 0.1 * np.linalg.norm(best_coef[0], 2)
        object_train_list.append(obj_train)
        object_test_list.append((obj_test))

        y_pred_train = (X_train @ best_coef[0]) + best_coef[1]
        y_pred_train = np.sign(y_pred_train)
        classification_rate_train = 1 - sum(y_pred_train == y_train) / len(y_train)
        classification_rate_train_list.append(classification_rate_train)

        y_pred_test = (X_test @ best_coef[0]) + best_coef[1]
        y_pred_test = np.sign(y_pred_test)
        classification_rate_test = 1 - sum(y_pred_test == y_test) / len(y_test)
        classification_rate_test_list.append(classification_rate_test)
        # print("classification_rate_train: ", classification_rate_train)
        # print("classification_rate_test: ", classification_rate_test)

    plt.plot(iteration_list, object_train_list, label = "Train")
    plt.plot(iteration_list, object_test_list, label = "Test")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.show()

    plt.plot(iteration_list, classification_rate_train_list, label = "Train")
    plt.plot(iteration_list, classification_rate_test_list, label = "Test")
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification Rate")
    plt.legend()
    plt.show()


