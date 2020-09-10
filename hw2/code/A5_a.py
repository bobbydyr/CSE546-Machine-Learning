from A4_A5_starter import *

if __name__ == '__main__':
    df_train = pd.read_csv("crime-train.txt", sep='\t')
    df_test = pd.read_csv("crime-test.txt", sep='\t')

    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 1].values.reshape((X_train.shape[0], 1))
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 1].values.reshape((X_test.shape[0], 1))
    n, d = X_train.shape

    lam = int(compute_initial_lamb(X_train, y_train))
    number_of_nonezero_feature = []
    loss_train_list = []
    loss_test_list = []
    selected_coef_history = np.zeros((20, 5))

    decrease_factor = 2
    lam_list = lam * (1/decrease_factor) ** np.arange(0, 20)
    householdsize_list = []
    agePct12t29_list = []
    agePct65up_list = []
    pctUrban_list = []
    pctWSocSec_list = []

    w_first = np.zeros((d,1))
    w_old = w_first[:]

    # lam_list = lam_list[-15:]
    # lam_list = [30]
    for i in range(len(lam_list)):
        print("############################################")
        lam = lam_list[i]
        print("lam", lam)
        lasso = Lasso(lamb=lam, delta=0.005)
        lasso.coordinate_descent(X_train, y_train)
        w_new = lasso.last_w.copy()
        print("Number of coe > 0:", np.count_nonzero(w_new))
        number_of_nonezero_feature.append(np.count_nonzero(w_new))

        loss_train = np.linalg.norm(X_train @ w_new - y_train, 2)
        loss_train_list.append(loss_train)
        loss_test = np.linalg.norm(X_test @ w_new - y_test, 2)
        loss_test_list.append(loss_test)
        # selected feature index: [1, 3, 5, 7, 12]
        householdsize_list.append(w_new[1])
        agePct12t29_list.append(w_new[3])
        agePct65up_list.append(w_new[5])
        pctUrban_list.append(w_new[7])
        pctWSocSec_list.append(w_new[12])

        w_old = w_new.copy()

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

        plt.plot(lam_list, selected_coef_history[i], label = feature)
        plt.xscale('log')
        plt.xlabel("Lambda")
    plt.legend(loc="upper left")
    plt.show()

    # A.5 c
    plt.plot(lam_list, loss_train_list, label = "Train")
    plt.plot(lam_list, loss_test_list, label = "Test")
    # plt.xscale('log')
    plt.xlabel("Lambda")
    plt.legend(loc="upper left")
    plt.show()


    coeff_data = pd.DataFrame({"Coeff": (list(w_new.T[0])),
                               "Name": df_train.columns[1:]}).sort_values(["Coeff"], ascending=False)


