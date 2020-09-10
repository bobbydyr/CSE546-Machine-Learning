from A4_A5_starter import *

if __name__ == '__main__':
    n = 500
    d = 1000
    k = 100

    X_train = generate_x(n, d)
    y_train, W_init = generate_y(n, d, k, X_train)
    lam = compute_initial_lamb(X_train, y_train)
    # lam_list = []
    number_of_nonezero_feature = []
    FDR_list = []
    TPR_list = []

    lam_list = lam * (1/1.5) ** np.arange(0, 20)

    for lam in lam_list:

        print("lam", lam)
        lasso = LASSO(lam, delta=0.4)
        lasso.coordinate_descent(X_train, y_train, np.zeros((d,1)))
        last_w = lasso.last_w.copy()
        print("Number of coe > 0:", sum(abs(last_w) > 0))
        number_nonezero = sum(last_w != 0)
        number_of_nonezero_feature.append(number_nonezero)


    plt.plot(lam_list, number_of_nonezero_feature)
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("# of none-zero coef")
    plt.show()
