import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def load_data():
    mndata = MNIST('python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test
X_train, y_train, X_test, y_test = load_data()

# One hot encoding
def one_hot(y_train_, m):
    n = len(y_train_)
    reformed_tensor = torch.zeros(n, m)
    for i in range(n):
        index = y_train_[i]
        reformed_tensor[i][index] = 1
    return reformed_tensor#

if __name__ == "__main__":
    # convert to tensor
    X_train_ = torch.tensor(X_train, dtype=torch.double)
    y_train_ = torch.tensor(y_train, dtype=torch.int64)

    X_test_ = torch.tensor(X_test, dtype=torch.double)
    y_test_ = torch.tensor(y_test, dtype=torch.int64)

    W = torch.zeros(784, 10, requires_grad=True, dtype=torch.double)
    W_mse = torch.zeros(784, 10, requires_grad=True, dtype=torch.double)
    step_size = 0.05
    epochs = 150
    train_accuracy_list = []
    test_accuracy_list = []
    train_accuracy_list_mse = []
    test_accuracy_list_mse = []

    epochs = list(range(epochs))
    for epoch in epochs:
        print("Epoch: ", epoch)
        y_hat = torch.matmul(X_train_, W)
        y_hat_mse = torch.matmul(X_train_, W_mse)
        # cross entropy combines softmax calculation with NLLLoss
        loss = torch.nn.functional.cross_entropy(y_hat, y_train_)
        loss_mse = torch.nn.functional.mse_loss(y_hat_mse, one_hot(y_train_, 10).double())
        # computes derivatives of the loss with respect to W
        loss.backward()
        loss_mse.backward()
        # gradient descent update
        W.data = W.data - step_size * W.grad
        W_mse.data = W_mse.data - step_size * W_mse.grad
        # .backward() accumulates gradients into W.grad instead
        # of overwriting, so we need to zero out the weights

        # Cross Entropy
        max_index_train = torch.max((torch.matmul(X_train_, W)), dim=1).indices.numpy()
        num_corrected_prediction_train = sum(max_index_train == y_train)
        train_accu = num_corrected_prediction_train / len(y_train)
        train_accuracy_list.append(train_accu)
        

        max_index_test = torch.max((torch.matmul(X_test_, W)), dim=1).indices.numpy()
        num_corrected_prediction_test = sum(max_index_test == y_test)
        test_accu = num_corrected_prediction_test / len(y_test)
        test_accuracy_list.append(test_accu)

        # MSE
        max_index_train_mse = torch.max((torch.matmul(X_train_, W_mse)), dim=1).indices.numpy()
        num_corrected_prediction_train_mse = sum(max_index_train_mse == y_train)
        train_accu_mse = num_corrected_prediction_train / len(y_train)
        train_accuracy_list_mse.append(train_accu_mse)

        max_index_test_mse = torch.max(torch.matmul(X_test_, W_mse), dim=1).indices.numpy()
        num_corrected_prediction_test_mse = sum(max_index_test_mse == y_test)
        test_accu_mse = num_corrected_prediction_test_mse / len(y_test)
        test_accuracy_list_mse.append(test_accu_mse)



        print("CROSS ENTOPY: Train Accuracy: ", train_accu)
        print("CROSS ENTOPY: Test Accuracy: ", test_accu)
        print("RIDGE: ", train_accu_mse)
        print("RIDGE: ", test_accu_mse)

        W.grad.zero_()
        W_mse.grad.zero_()

    plt.plot(epochs, train_accuracy_list, label="Train")
    plt.plot(epochs, test_accuracy_list, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy - Cross Entropy")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw2/latex/B4_c_1.png")
    plt.show()

    plt.plot(epochs, train_accuracy_list_mse, label="Train")
    plt.plot(epochs, test_accuracy_list_mse, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy - Ridge Regression")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw2/latex/B4_c_2.png")
    plt.show()
















