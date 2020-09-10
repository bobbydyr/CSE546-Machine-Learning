# A3.d - 1 #################################################################################
import numpy as np
import matplotlib.pyplot as plt

n = 300
# np.random.seed(1)

x = np.random.uniform(0,1,n)
y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2)) + np.random.standard_normal(n)

def k_poly(x, z, d):
    a = x @ z.T
    k = (1 + x @ z.T)**d
    return k

error_validation_list = []
lamb = 500
cv_size = int(n/10)
lamb_list = []
d_list = []
for lamb in list(500 * (1/2)**(np.arange(0, 20))):
    for d in list(range(0, 51)):
        error_validation = 0
        print("Lam: ", lamb, ", d: ", d)
        for i in range(0, n, cv_size):
            x_train = np.append(x[0:i], x[i+cv_size:n])
            y_train = np.append(y[0:i], y[i+cv_size:n])
            x_validation = x[i:i+cv_size]
            y_validation = y[i:i+cv_size]
            K = k_poly(x_train[:, np.newaxis], x_train[:, np.newaxis], d)
            alpha = np.linalg.pinv(K + lamb) @ y_train
            # in predicted y formula
            k_xi_x = (1 + np.multiply(x_validation, np.repeat(x_train[np.newaxis, :], cv_size).reshape(270,cv_size)))**d
            # y_predicted = alpha @ k_xi_x.T
            y_predicted = alpha[np.newaxis, :] @ k_xi_x
            error_validation += np.sum((y_predicted - y_validation).T @ (y_predicted- y_validation))
            # error_validation = error_validation[0][0]
        error_validation /= n
        print("error_validation: ", error_validation)
        error_validation_list.append(error_validation)
        lamb_list.append(lamb)
        d_list.append(d)

# min_error = min(error_validation_list)
index_boostrap_sample_min_error = error_validation_list.index(min(error_validation_list))
lamb_best_poly = lamb_list[index_boostrap_sample_min_error]
d_best = d_list[index_boostrap_sample_min_error]
print("Best lamb: ", lamb_best_poly, ", Best d: ", d_best)

# Best lamb:  0.003814697265625 , Best d:  40

# plots the comparaison
np.random.seed(1)
n = 100
x_fine = np.array(list(range(0, 100, 1))) / 100
y_fine_true = 4*np.sin(np.pi*x_fine)*np.cos(6*np.pi*(x_fine**2))
y_fine_grid = y_fine_true + np.random.standard_normal(n)
f_poly_predicted = []
for xi in x_fine:
    K = k_poly(x_fine[:, np.newaxis], x_fine[:, np.newaxis], d_best)
    alpha = np.linalg.pinv(K + lamb_best_poly) @ y_fine_grid
    k_xi_x = (1 + xi * x_fine[np.newaxis, :]) ** d_best  # use this when polynomial kernel
    y_predicted = alpha @ k_xi_x.T

    f_poly_predicted.append(y_predicted)

plt.plot(x_fine, y_fine_true, label='True')
plt.plot(x_fine, f_poly_predicted, label='Poly Kernel')
plt.plot(x, y,'bo', label='Observed')
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3d_1_test.png")
plt.show()

# -------------------------------- #

B = 300
n=300
m = 1000
x_fine = np.arange(min(x),max(x),0.01)
n_fine = len(x_fine)
# np.random.seed(10)
boostrap_predicted_poly_matrix = []
x = np.random.uniform(0,1,n)
y_true_sample = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2))
y_observed = y_true_sample + np.random.randn(n)

for j in range(B):
    index_boostrap_sample = np.random.choice(n,n)
    x_training = x[index_boostrap_sample]
    y_training = y_observed[index_boostrap_sample]
    K = k_poly(x_training[:,np.newaxis],x_training[:,np.newaxis], d_best)
    alpha = np.linalg.solve((K + lamb_best_poly*np.eye(n, n)), y_training)
    y_predicted_boostrap_ploy = []
    # for xi in np.array(list(range(0, 100, 1))) / 100:
    # test
    K = k_poly(x_training[:,np.newaxis],x_training[:,np.newaxis], d_best)
    alpha = np.linalg.solve((K + lamb_best_poly*np.eye(n, n)), y_training)

    for xi in x_fine:
        y_predicted_boostrap_ploy.append(np.sum((1+xi*x_training[np.newaxis,:]) ** d_best @ alpha))
    boostrap_predicted_poly_matrix.append(y_predicted_boostrap_ploy)
boostrap_predicted_poly_matrix = np.array(boostrap_predicted_poly_matrix)

percent_5_list_poly = []
percent_95_list_poly = []
for i in range(n_fine):
    sorted_xi_from_300_B_sample = np.sort(boostrap_predicted_poly_matrix[:, i])
    x_percentile_5 = sorted_xi_from_300_B_sample[int(B * 0.05)]
    x_percentile_95 = sorted_xi_from_300_B_sample[int(B * 0.95)]
    percent_5_list_poly.append(x_percentile_5)
    percent_95_list_poly.append(x_percentile_95)

# x_fine = np.array(list(range(0, 100, 1))) / 100
y_fine_true = 4*np.sin(np.pi*x_fine)*np.cos(6*np.pi*(x_fine**2))
plt.plot(x_fine, y_fine_true, label = 'True Model')
plt.plot(np.array(list(range(0, 100, 1))) / 100, f_poly_predicted, label = 'Poly Kernel Prediction')
plt.fill_between(x_fine, percent_5_list_poly, percent_95_list_poly, alpha=0.4, label="90% CI")
plt.plot(x, y_observed,'bo', alpha=0.2, label ='Observed data')
plt.ylim(-6, 6)
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3e_1_test.png")
plt.show()







###################################################################################################################

# A3.d - 2

def k_rbf(x, z, gamma):
    return np.exp(-gamma*(x-z)*(x-z))

n = 300
np.random.seed(1)
cv_size = int(n/10)
x = np.random.uniform(0,1,n)
y_true = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2))
y = y_true + np.random.randn(n)
error_validation_list = []
lamb_list = []
gamma_list = []
d_list =[]

lamb = 1
for lamb in list(500 * (1/2)**(np.arange(0,30))):
    for gamma in list(50 * (1/1.1)**(np.arange(0,30))):
        print("Lam: ", lamb, ", gamma: ", gamma)
        error_validation = 0
        for i in range(0, n, cv_size):
            x_train = np.append(x[0:i], x[i+cv_size:n])
            y_train = np.append(y[0:i], y[i+cv_size:n])
            x_validation = x[i:i+cv_size]
            y_validation = y[i:i+cv_size]
            K = k_rbf(x_train[:,np.newaxis],x_train[np.newaxis,:], gamma)
            alpha = np.linalg.pinv(K + lamb) @ y_train
            k_xi_x = np.exp(-gamma*(x_validation - np.repeat(x_train[np.newaxis,:], cv_size).reshape((n-cv_size, cv_size)))**2)
            y_predicted = alpha[np.newaxis, :] @ k_xi_x
            error_validation += np.sum((y_predicted - y_validation).T @ (y_predicted- y_validation))
        error_validation /= n
        error_validation_list.append(error_validation)
        print("error_validation: ", error_validation)
        lamb_list.append(lamb)
        gamma_list.append(gamma)

min_error = min(error_validation_list)
index_boostrap_sample_min_error = error_validation_list.index(min_error)
lamb_best_rbf = lamb_list[index_boostrap_sample_min_error]
gamma_best = gamma_list[index_boostrap_sample_min_error]
print('Best gamma for RBF kernel is : ', gamma_best)
print('Best Lambda for RBF kernel is :', lamb_best_rbf)

# Best gamma for RBF kernel is :  8.992939495460687
# Best Lambda for RBF kernel is : 1.862645149230957e-06
# plots the comparaison
n = 100
np.random.seed(10)

x_fine = np.array(list(range(0, 100, 1))) / 100
y_fine_true = 4*np.sin(np.pi*x_fine)*np.cos(6*np.pi*(x_fine**2))
y_fine_grid = y_fine_true + np.random.standard_normal(n)

f_rbf_predicted = []
K_rbf = k_rbf(x_fine[:,np.newaxis],x_fine[np.newaxis,:], gamma_best)
alpha = np.linalg.solve((K_rbf + lamb_best_rbf*np.eye(n, n)), y_fine_grid)
for xi in x_fine:
    f_rbf_predicted.append(np.sum(alpha * np.exp(-gamma_best*(xi-x_fine)**2)))

plt.plot(x_fine, y_fine_true, label = 'True Model')
plt.plot(x_fine, f_rbf_predicted, label = 'RBF Kernel Prediction')
plt.plot(x, y,'bo', label ='Observed data')
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3d_2_test.png")
plt.show()


# A3.e2
B = 300
n=300
# m=1000
# n_fine = 100
np.random.seed(0)
boostrap_predicted_rbf_matrix = []
# x = np.array(list(range(0, 100, 1))) / 100
x = np.random.uniform(0,1,n)
y_true_sample = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2))
y_observed = y_true_sample + np.random.randn(n)

for j in range(B):
    index_boostrap_sample = np.random.choice(n,n)
    x_training = x[index_boostrap_sample]
    y_training = y_observed[index_boostrap_sample]
    K_rbf = k_rbf(x_training[:, np.newaxis], x_training[np.newaxis, :], gamma_best)
    alpha = np.linalg.solve((K_rbf + lamb_best_rbf * np.eye(n, n)), y_training)

    y_predicted_boostrap_rbf = []
    for xi in x_fine:
        # k_xi_x = np.exp(-gamma * (xi - x_training[np.newaxis, :]) ** 2)
        # predicted = k_xi_x@alpha
        y_predicted_boostrap_rbf.append(np.sum(alpha * np.exp(-gamma_best*(xi-x_training)**2)))
    boostrap_predicted_rbf_matrix.append(y_predicted_boostrap_rbf)
boostrap_predicted_rbf_matrix = np.array(boostrap_predicted_rbf_matrix)

percent_5_list_rbf = []
percent_95_list_rbf = []
for i in range(len(x_fine)):
    sorted_xi_from_300_B_sample = np.sort(boostrap_predicted_rbf_matrix[:, i])
    x_percentile_5 = sorted_xi_from_300_B_sample[int(B * 0.05)]
    x_percentile_95 = sorted_xi_from_300_B_sample[int(B * 0.95)]
    percent_5_list_rbf.append(x_percentile_5)
    percent_95_list_rbf.append(x_percentile_95)

x_fine = np.array(list(range(0, 100, 1))) / 100
y_fine_true = 4*np.sin(np.pi*x_fine)*np.cos(6*np.pi*(x_fine**2))
plt.plot(x_fine, y_fine_true, label = 'True Model')
plt.plot(x_fine, f_rbf_predicted, label = 'rbf Kernel Prediction')
plt.fill_between(x_fine, percent_5_list_rbf, percent_95_list_rbf, alpha=0.4, label="90% CI")
plt.plot(x, y_observed,'bo', alpha=0.2, label ='Observed data')
plt.ylim(-6, 6)
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3e_2_test.png")
plt.show()







########################################## A3 e ##########################################
#
#                                          A3 e
#
########################################## A3 e ##########################################
d_best=33

B = 300
n=300
m=1000
np.random.seed(4)
x = np.random.uniform(0,1,m+n)
y_true_sample = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2))
y_observed = y_true_sample + np.random.randn(m+n)
error_difference_list = np.zeros(B)

for j in range(B):
    print("B: ", j)
    index_boostrap_sample = np.random.choice(m+n,m)
    x_training = x[index_boostrap_sample]
    y_training = y_observed[index_boostrap_sample]
    # comput poly kernel
    K = k_poly(x_training[:, np.newaxis], x_training[:, np.newaxis], d_best)
    alpha = np.linalg.pinv(K + lamb_best_poly) @ y_training
    # compute rbf kernel
    K_rbf = k_rbf(x_training[:, np.newaxis], x_training[np.newaxis, :], gamma_best)
    alpha_rbf = np.linalg.solve((K_rbf + lamb_best_rbf * np.eye(m, m)), y_training)

    error_difference = 0
    for i in range(len(x_training)):
        # poly predicted
        k_xi_x = (1 + x_training[i] * x_training[np.newaxis, :]) ** d_best  # use this when polynomial kernel
        y_predicted = alpha @ k_xi_x.T
        # rbf predicted
        y_predicted_rbf = np.sum(alpha_rbf * np.exp(-gamma_best*(x_training[i]-x_training)**2))
        # error difference
        error_difference += (y_training[i] - y_predicted)**2 - (y_training[i] - y_predicted_rbf)**2
    error_difference /= len(x_training)
    # error_difference_list.append(error_difference)
    print("error_difference: ", error_difference)
    error_difference_list[j] = error_difference
    error_difference_list = np.sort(error_difference_list)

print("5%: ", error_difference_list[int(B * 0.05)])
print("95%: ", error_difference_list[int(B * 0.95)])
