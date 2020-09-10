import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        W = initial_w
        A = 2 * np.sum(np.power(X, 2), axis=0)
        loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(abs(W))

        not_converge = True
        while not_converge:
            self.b = np.average(y - X.dot(W))
            loss_prev = loss
            prev_w = np.copy(W)
            for k in range(d):
                X_k = X[:, k]
                prev_wk = np.copy(W[k])
                W[k] = 0
                c_k = np.dot((y - (self.b * np.ones((n, 1)) + X.dot(W))).T, X_k)
                if 2 * c_k + self.lamb < 0:
                    W[k] = (c_k * 2 + self.lamb) / A[k]
                elif 2 * c_k - self.lamb > 0:
                    W[k] = (c_k * 2 - self.lamb) / A[k]
                else:
                    W[k] = 0

            if sum(abs(W - prev_w)) <= sum(abs(self.delta * prev_w)):
                not_converge = False
                print(W)

            loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(abs(W))
            self.loss_list.append(loss)
            self.last_w = W
            self.last_selected_coef = self.last_w.T[0][self.selected_feature_index]

    def predict(self, X):
        return X.dot(self.last_w)



def compute_initial_lamb(x, y):
    n, d = x.shape
    lamb_array = np.zeros((d, 1))
    for k in range(d):
        lamb_temp = 2 * abs(x[:, k].T @ (y - np.mean(y)))
        lamb_array[k] = lamb_temp
    return max(lamb_array)

if __name__ == "__main__":
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")
    y_train = df_train["ViolentCrimesPerPop"].values.reshape(df_train.shape[0], 1)
    x_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
    x_test = df_test.drop("ViolentCrimesPerPop", axis=1).values
    y_test = df_test["ViolentCrimesPerPop"].values.reshape(df_test.shape[0], 1)
    print("data import done")


    n ,d = x_train.shape
    lambda_max = compute_initial_lamb(x_train, y_train)
    initial_model = LASSO(lambda_max)
    initial_model.coordinate_descent(x_train, y_train, np.zeros((d, 1)))
    initial_w = initial_model.last_w

    lam_list = (lambda_max) * (1 / 2) ** np.arange(0, 20)
    householdsize_list = []
    agePct12t29_list = []
    agePct65up_list = []
    pctUrban_list = []
    pctWSocSec_list = []
    trained_w = []
    initial_loss_train = np.mean((initial_model.predict(x_train) - y_train)**2)
    initial_loss_test = np.mean((initial_model.predict(x_test) - y_test)**2)

    loss_train_list = [initial_loss_train]
    loss_test_list = [initial_loss_test]
    number_of_nonezero_feature = []

    # lam_list = [30,30]
    for lam in lam_list[1:]:
        model = LASSO(lam)
        model.coordinate_descent(x_train, y_train, initial_w)
        w_new = model.last_w.copy()
        trained_w.append(w_new)
        number_of_nonezero_feature.append(np.count_nonzero(w_new))
        householdsize_list.append(w_new[1])
        agePct12t29_list.append(w_new[3])
        agePct65up_list.append(w_new[5])
        pctUrban_list.append(w_new[7])
        pctWSocSec_list.append(w_new[12])

        loss_train = np.mean((model.predict(x_train) - y_train)**2)
        loss_train_list.append(loss_train)
        loss_test = np.mean((model.predict(x_test) - y_test)**2)
        loss_test_list.append(loss_test)

    selected_coef_history = [householdsize_list, agePct12t29_list, agePct65up_list, pctUrban_list, pctWSocSec_list]

    # A.5 a
    # Plot lambda against number_of_nonezero_feature
    plt.plot(lam_list[1:], number_of_nonezero_feature)
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("# of none-zero coef")
    plt.show()

    # A.5 b
    # plot 5 different feature change with different lambda
    features = ['householdsize', 'agePct12t29', 'agePct65up', 'pctUrban', 'pctWSocSec']
    for i, feature in enumerate(features):
        plt.plot(lam_list[1:], selected_coef_history[i], label=feature)
        plt.xscale('log')
        plt.xlabel("Lambda")
        plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()

    # A.5 c
    plt.plot(lam_list, loss_train_list, label="Train")
    plt.plot(lam_list, loss_test_list, label="Test")
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()

    coeff_data = pd.DataFrame({"Coeff": (list(w_new.T[0])),
                               "Name": df_train.columns[1:]}).sort_values(["Coeff"], ascending=False)
    coeff_data[:10]
    coeff_data[-10:]
