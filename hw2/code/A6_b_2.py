import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

np.random.seed(666)
L = 0.1


def load_data():
    mndata = MNIST('python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    labels_train = labels_train.astype(np.int16)
    labels_test = labels_test.astype(np.int16)
    X_train = np.vstack( (X_train[labels_train == 2], X_train[labels_train == 7]))
    y_train = np.hstack((labels_train[labels_train == 2], labels_train[labels_train == 7]))
    y_train[y_train == 2] = -1
    y_train[y_train == 7] = 1

    X_test = np.vstack((X_test[labels_test == 2], X_test[labels_test == 7]))
    y_test = np.hstack((labels_test[labels_test == 2], labels_test[labels_test == 7]))
    y_test[y_test == 2] = -1
    y_test[y_test == 7] = 1

    print(X_train.shape, y_train.shape, y_train.sum())
    return (X_train, y_train, X_test, y_test)


def descent(X, Y, w, b, eta=0.5):
    # update b
    u = 1.0 / (1.0 + np.exp(-Y * (b + X.dot(w))))
    gradient_b = (-Y * (1 - u)).mean()
    b -= eta * gradient_b

    # update w
    u = 1.0 / (1.0 + np.exp(-Y * (b + X.dot(w))))
    xy = np.multiply(X.T, Y)
    gradient_w = (- xy * (1 - u)).mean(axis=1) + 2 * L * w
    w -= eta * gradient_w

    return (w, b)


def objective(X, Y, w, b):
    inside = np.log(1.0 + np.exp(-Y * (b + X.dot(w))))
    obj_value = inside.mean() + L * np.linalg.norm(w, 2)

    predicted = b + X.dot(w)
    predicted[predicted < 0] = -1
    predicted[predicted >= 0] = 1
    correct = np.sum(predicted == Y)
    # print(correct)
    error = 1.0 - float(correct) / float(X.shape[0])

    return (obj_value, error)


def run(X_train, Y_train, X_test, Y_test, eta, itersize, batch=0, save_plt_name="A6"):
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0

    iters = []
    test_j = []
    train_j = []
    test_e = []
    train_e = []
    j, error = objective(X_train, Y_train, w, b)
    tj, terror = objective(X_test, Y_test, w, b)
    test_j.append(tj)
    train_j.append(j)
    test_e.append(terror)
    train_e.append(error)
    iters.append(0)

    i = 1
    for dataloop in range(0, itersize):
        n, d = X_train.shape
        index = np.random.permutation(n)  # Random Sampling
        X_train = X_train[index]
        Y_train = Y_train[index]
        split = n / batch
        Xs = np.array_split(X_train, split)
        Ys = np.array_split(Y_train, split)

        # compute and update weights for batch size
        for X_split, Y_split in zip(Xs, Ys):
            w, b = descent(X_split, Y_split, w, b, eta=eta)
            j, error = objective(X_train, Y_train, w, b)
            tj, terror = objective(X_test, Y_test, w, b)

            test_j.append(tj)
            train_j.append(j)
            test_e.append(terror)
            train_e.append(error)
            iters.append(i)
            if (i % 100 == 0 or split == 1):
                print(j, error, i, dataloop, X_split.shape)
            i += 1

    plt.plot(iters, test_j, label="Test")
    plt.plot(iters, train_j, label="Train")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw2/latex/"+ save_plt_name + "_1.png")
    plt.show()

    plt.plot(iters, test_e, label="Test")
    plt.plot(iters, train_e, label="Train")
    plt.xlabel("Iteration")
    plt.ylabel("Misclassified Rate")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw2/latex/"+ save_plt_name + "_2.png")
    plt.show()

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    n, d = X_train.shape
    print("#######  Gradient Descent #######")
    run(X_train, Y_train, X_test, Y_test, 0.5, 50, batch=n, save_plt_name="A6_b")
    print("#######  Stochastic Gradient Descent  #######")
    run( X_train, Y_train, X_test, Y_test, 0.001, 1, batch=1, save_plt_name="A6_c")
    print("#######  Mini Batch Gradient Descent  #######")
    run( X_train, Y_train, X_test, Y_test, 0.01, 10, batch=100, save_plt_name="A6_d")






