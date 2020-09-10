import numpy as np
import torch
import matplotlib.pyplot as plt
from mnist import MNIST

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

# One hot encoding
def one_hot(y_train_, m):
    n = len(y_train_)
    reformed_tensor = torch.zeros(n, m)
    for i in range(n):
        index = y_train_[i]
        reformed_tensor[i][index] = 1
    return reformed_tensor

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(size=1)
    print("Load Data Compelete!")
    # convert to tensor
    dtype = torch.FloatTensor

    X_train_ = torch.tensor(X_train, dtype=torch.double).type(dtype)
    y_train_ = torch.tensor(y_train, dtype=torch.int64)

    X_test_ = torch.tensor(X_test, dtype=torch.double).type(dtype)
    y_test_ = torch.tensor(y_test, dtype=torch.int64)

    def ReLU(x):
        return abs(x) * (x > 0)

    def model(x, w0, w1, b0, b1):
        return (w1 @ ReLU(w0 @ x.T + b0) + b1).T

    def alpha(d):
        return 1 / np.sqrt(d)
    h = 64
    k = 10
    n_train, d_train = X_train.shape
    w0_data = (alpha(d_train) + alpha(d_train))*(torch.rand(h, d_train) -alpha(d_train)).type(dtype)
    w0 = torch.autograd.Variable(w0_data, requires_grad=True)
    b0_data = (alpha(d_train) + alpha(d_train))*(torch.rand(h, 1) -alpha(d_train)).type(dtype)
    b0 = torch.autograd.Variable(b0_data,requires_grad=True)
    w1_data = (alpha(d_train) + alpha(d_train))*(torch.rand(k, h) -alpha(d_train)).type(dtype)
    n_test, d_test = X_test.shape
    w1 = torch.autograd.Variable(w1_data,requires_grad=True)
    b1_data = (alpha(d_train) + alpha(d_train))*(torch.rand(k, 1) -alpha(d_train)).type(dtype)
    b1 = torch.autograd.Variable(b1_data, requires_grad=True)
    step_size = 0.005
    epochs = 150
    train_accuracy_list = []
    test_accuracy_list = []
    loss_train_list = []
    loss_test_list = []

    optim = torch.optim.Adam([w0, w1, b0, b1], lr=step_size)

    train_accu = 0
    test_accu = 0
    epochs_list = list(range(epochs))
    # for epoch in epochs_list:
    iter = 0
    while train_accu < 0.99:
        iter += 1
        # print("Epoch: ", epoch)
        optim.zero_grad()
        y_hat = model(X_train_, w0, w1, b0, b1)
        y_hat_index = torch.max(y_hat, dim=0).indices
        loss = torch.nn.functional.cross_entropy(y_hat, y_train_)
        loss.backward()
        optim.step()
        # Cross Entropy
        max_index_train = torch.max(model(X_train_, w0, w1, b0, b1), dim=1).indices.numpy()
        num_corrected_prediction_train = sum(max_index_train == y_train)
        train_accu = num_corrected_prediction_train / len(y_train)
        train_accuracy_list.append(train_accu)
        loss_train_list.append(loss)


        max_index_test = torch.max(model(X_test_, w0, w1, b0, b1), dim=1).indices.numpy()
        num_corrected_prediction_test = sum(max_index_test == y_test)
        test_accu = num_corrected_prediction_test / len(y_test)
        test_accuracy_list.append(test_accu)
        loss_test_list.append(torch.nn.functional.cross_entropy(model(X_test_, w0, w1, b0, b1), y_test_))
        print("##################", iter, "##################")
        print("CROSS ENTROPY: Train Accuracy: ", train_accu)
        print("CROSS ENTROPY: Test Accuracy: ", test_accu)

    print("CROSS ENTROPY: Train Accuracy: ", train_accuracy_list[-1])
    print("CROSS ENTROPY: Test Accuracy: ", test_accuracy_list[-1])
    print("Train Loss: ", loss_train_list[-1])
    print("Test Loss: ", loss_test_list[-1])

    plt.plot(range(iter), train_accuracy_list, label="Train")
    plt.plot(range(iter), test_accuracy_list, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A5a.png")
    plt.show()

    plt.plot(range(iter), loss_train_list, label="Train")
    plt.plot(range(iter), loss_test_list, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Loss - Cross Entropy")
    plt.legend()
    plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A5a_2.png")
    plt.show()

    w0.shape[0] * w0.shape[1] + w1.shape[0] * w1.shape[1] + len(b0) + len(b1)

# CROSS ENTROPY: Train Accuracy:  0.99
# CROSS ENTROPY: Test Accuracy:  0.9701
# Train Loss:  tensor(0.0408, grad_fn=<NllLossBackward>)
# Test Loss:  tensor(0.1029, grad_fn=<NllLossBackward>)
#####################################################################################################################
