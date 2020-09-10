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


    def fit(self, X, y, initial_w):
        n, d = X.shape
        # W = np.zeros((d, 1))
        W = initial_w
        A = 2 * np.sum(np.power(X, 2), axis=0)  # Power and then sum up the rows # C = np.zeros((d, 1))
        # Iterate until convergence
        loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(
            abs(W))

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
                # Check if converged
                # if abs(W[k] - prev_wk) > self.delta:
                # converged = False
            if sum(abs(W - prev_w)) <= sum(abs(self.delta * prev_w)):
                converged = True
                print(W)
            # Sanity Check
            loss = np.sum(np.power((self.b * np.ones((n, 1)) + X.dot(W) - y), 2)) + self.lamb * np.sum(abs(W))
            self.loss_list.append(loss)
            print(loss)
            self.last_w = W
            self.last_selected_coef = self.last_w.T[0][self.selected_feature_index]

    def predict(self, X):
        return X.dot(self.last_w)

    def num_of_nonzeros(self):
        return np.count_nonzero(self.last_w)


def create_lambdas(init_lambda, ratio, num):
    return init_lambda * (1 / ratio) ** np.arange(0, num)

if __name__ == "__main__":  # Importing Data
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")
    y_train = df_train["ViolentCrimesPerPop"].values.reshape(df_train.shape[0], 1)
    x_train = df_train.drop("ViolentCrimesPerPop", axis = 1).values
    x_test = df_test.drop("ViolentCrimesPerPop", axis = 1).values
    y_test = df_test["ViolentCrimesPerPop"].values.reshape(df_test.shape[0], 1)
    print("data import done")
    # Computing
    lambda_ratio = 2
    lambda_num = 20
    lambda_max = np.max(2 * np.abs(x_train.T.dot(y_train - np.average(y_train))))
    lambda_max_model = LASSO(lambda_max)
    lambda_max_model.fit(x_train, y_train)
    lambda_max_w = lambda_max_model.last_w

    print("lambda_max is ", lambda_max)
    lambdas = create_lambdas(lambda_max, lambda_ratio, lambda_num)
    print("lambdas are ", lambdas)
    models = [LASSO(lamb) for lamb in lambdas]
    for model in models:
        model.fit(x_train, y_train)

    # a
    number_of_nonezero_feature = [model.num_of_nonzeros() for model in models]


    lambda_max = np.max(2 * np.abs(x_train.T.dot(y_train - np.average(y_train))))
    lambda_max_model = LASSO(lambda_max)
    lambda_max_model.fit(x_train, y_train)
    lambda_max_w = lambda_max_model.last_w

    lam_list = lambda_max * (1 / 2) ** np.arange(0, 20)
    householdsize_list = []
    agePct12t29_list = []
    agePct65up_list = []
    pctUrban_list = []
    pctWSocSec_list = []
    trained_w = []
    loss_train_list = []
    loss_test_list = []
    for i in lambdas:
        model = LASSO(lambdas[i], lambda_max_w)
        model.fit(x_train, y_train)
        w_new = model.last_w
        trained_w.append(w_new)
        number_of_nonezero_feature.append(model.num_of_nonzeros())
        householdsize_list.append(w_new[1])
        agePct12t29_list.append(w_new[3])
        agePct65up_list.append(w_new[5])
        pctUrban_list.append(w_new[7])
        pctWSocSec_list.append(w_new[12])

        loss_train = np.linalg.norm(x_train @ w_new - y_train, 2)
        loss_train_list.append(loss_train)
        loss_test = np.linalg.norm(x_train @ w_new - y_test, 2)
        loss_test_list.append(loss_test)

    selected_coef_history = [householdsize_list, agePct12t29_list, agePct65up_list, pctUrban_list, pctWSocSec_list]


    # A.5 a
    # Plot lambda against number_of_nonezero_feature
    plt.plot(lam_list, number_of_nonezero_feature)
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("# of none-zero coef")
    plt.show()

    # A.5 b
    # plot 5 different feature change with different lambda
    features = ['householdsize', 'agePct12t29', 'agePct65up', 'pctUrban', 'pctWSocSec']
    for i, feature in enumerate(features):

        plt.plot(lam_list, selected_coef_history[i], label=feature)
        plt.xscale('log')
        plt.xlabel("Lambda")
    plt.legend(loc="upper left")
    plt.show()

    # A.5 c
    plt.plot(lam_list, loss_train_list, label = "train squared error")
    plt.plot(lam_list, loss_test_list, label = "Test squared error")
    # plt.xscale('log')
    plt.xlabel("lambda")
    plt.legend(loc="upper left")
    plt.show()


    coeff_data = pd.DataFrame({"Coeff": (list(w_new.T[0])),
                               "Name": df_train.columns[1:]}).sort_values(["Coeff"], ascending=False)





















    trained_w = [model.last_w for model in models]
    print("lambdas non-zeros are", num_of_nonzeros)
    # Plot nonzeros
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("num of non-zeros")
    plt.plot(lambdas, num_of_nonzeros)
    plt.legend()
    plt.show()
    # b Plot reg path
    agePct12t29_index = df_train.columns.get_loc("agePct12t29")
    pctWSocSec_index = df_train.columns.get_loc("pctWSocSec")
    pctUrban_index = df_train.columns.get_loc('pctUrban')
    agePct65up_index = df_train.columns.get_loc("agePct65up")
    householdsize_index = df_train.columns.get_loc('householdsize')
    agePct12t29_coeffs = [item[agePct12t29_index] for item in trained_w]
    print(agePct12t29_coeffs)
    pctWSocSec_coeffs = [item[pctWSocSec_index] for item in trained_w]
    pctUrban_coeffs = [item[pctUrban_index] for item in trained_w]
    agePct65up_coeffs = [item[agePct65up_index] for item in trained_w]
    householdsize_coeffs = [item[householdsize_index] for item in trained_w]
    plt.plot(lambdas, agePct12t29_coeffs, label="agePct12t29")
    plt.plot(lambdas, pctWSocSec_coeffs, label="pctWSocSec")
    plt.plot(lambdas, agePct65up_coeffs, label="agePct65up")
    plt.plot(lambdas, pctUrban_coeffs, label="pctUrban")
    plt.plot(lambdas, householdsize_coeffs, label="householdsize")
    plt.xlabel("lambda")
    plt.xscale('log')
    plt.ylabel("coefficient")
    plt.legend()
    plt.show()
    # c Plot Errors
    y_train_predicted = np.asarray([model.predict(x_train) for model in models])
    y_test_predicted = np.asarray([model.predict(x_test) for model in models])
    train_squared_error = np.average(np.power(y_train_predicted - y_train, 2), axis=1).squeeze(axis=1)
    test_squared_error = np.average(np.power(y_test_predicted - y_test, 2), axis=1).squeeze(axis=1)
    plt.plot(lambdas, train_squared_error, label="training squared error")
    plt.plot(lambdas, test_squared_error, label="test squared error")
    plt.xlabel("lambda")
    plt.xscale('log')
    plt.ylabel("squared error")
    plt.legend()
    plt.show()
    # d
    test_lambda = 30
    test_model = LASSO(test_lambda)
    test_model.fit(x_train, y_train)
    print("training complete")
    test_w = test_model.w.squeeze()
    top_positive_indices = test_w.argsort()[-1:][::-1]
    top_negative_indices = test_w.argsort()[:1][::-1]
    print("top_positive_indices is", top_positive_indices, " with name ", df_train.columns[top_positive_indices + 1],
          "value is ", test_w[top_positive_indices])
    print("top_negative_indices is", top_negative_indices, " with name ", df_train.columns[top_negative_indices + 1],
          "value is ", test_w[top_negative_indices])
