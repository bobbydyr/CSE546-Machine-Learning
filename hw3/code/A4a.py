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
X_train, y_train, X_test, y_test = load_data(size=0.1)
print("Load Data Compelete!")


def compute_cluster(X_train, centers, k):
    objective_value = 0
    clusters = [[] for i in range(k)]
    for i in np.arange(len(X_train)):
        distance_list = []
        for center in centers:
            norm = np.linalg.norm(X_train[i] - center)
            distance_list.append(norm)
        closesed_center_index = distance_list.index(min(distance_list))
        objective_value += min(distance_list)**2
        clusters[closesed_center_index].append(X_train[i])
    return clusters, objective_value


def compute_centers(classes):
    centers = []
    for i in range(len(classes)):
        centers.append(np.mean(classes[i], axis = 0))
    return centers


objective_value_list = []

k = 10
objs = []
iteration = 0
old_centers = random.sample(list(X_train), k)
new_centers = random.sample(list(X_train), k)
while not np.array_equal(old_centers, new_centers):
    iteration += 1
    print("Iteration: ", iteration)
    print("----------  Compute Clusters -----------")
    old_centers = new_centers
    clusters, objective_value = compute_cluster(X_train, new_centers, k)
    print("objective_value: ", objective_value)
    new_centers = compute_centers(clusters)
    objective_value_list.append(objective_value)

