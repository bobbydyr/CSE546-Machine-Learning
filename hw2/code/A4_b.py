from A4_A5_starter import *

if __name__ == '__main__':
    n = 500
    d = 1000
    k = 100

    X_train = generate_x(n, d)
    y_train, W_init = generate_y(n, d, k, X_train)
    lam = compute_initial_lamb(X_train, y_train)[0]
    # lam_list = []
    number_of_nonezero_feature = []
    FDR_list = []
    TPR_list = []

    lam_list = lam * (1/1.5) ** np.arange(0, 40)

    for lam in lam_list:

        print("lam", lam)
        lasso = LASSO(lam, delta=0.001)
        lasso.coordinate_descent(X_train, y_train, np.zeros((d,1)))
        last_w = lasso.last_w
        print("Number of coe > 0:", sum(abs(last_w) > 0))
        number_nonezero = sum(last_w != 0)
        number_of_nonezero_feature.append(number_nonezero)

        incorrect_none_zero = sum(last_w[W_init == 0] != 0)
        number_correct_none_zero = sum(last_w[W_init != 0] != 0)
        if incorrect_none_zero == 0:
            FDR = 0
            FDR_list.append(0)
        else:
            FDR = incorrect_none_zero / number_nonezero
            FDR_list.append(FDR)
        TPR = number_correct_none_zero / k
        TPR_list.append(TPR)

        print("FDR: ", FDR, " TPR: ", TPR)



    plt.plot(FDR_list, TPR_list)
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.show()
