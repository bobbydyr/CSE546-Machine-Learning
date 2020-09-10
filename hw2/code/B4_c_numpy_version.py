import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def load_data():
    mndata = MNIST('python-mnist/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    labels_train = labels_train.astype(np.int16)
    labels_test = labels_test.astype(np.int16)

    return (X_train, labels_train, X_test, labels_test)



