# A3.b1
import numpy as np
import matplotlib.pyplot as plt

n = 30
# np.random.seed(1)
x = np.random.uniform(0,1,n)
x_mean = np.mean(x)
x_sd = np.std(x)
# x = (x-x_mean)  # x after standardization

y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*(x**2)) + np.random.standard_normal(n)
# y = (y - x_mean) / x_sd
def k_poly(x, z, d):
    a = x @ z.T
    k = (1 + x @ z.T)**d
    return k

error_validation_list = []
lamb = 500
lamb_list = []
d_list = []
for lamb in list(500 * (1/2)**(np.arange(0,20))):
    for d in list(range(0, 51)):
        error_validation = 0
        print("Lam: ", lamb, ", d: ", d)
        for i in range(n):
            x_train = np.append(x[0:i], x[i+1:n])
            y_train = np.append(y[0:i], y[i+1:n])
            x_validation = x[i]
            y_validation = y[i]
            K = k_poly(x_train[:, np.newaxis], x_train[:, np.newaxis], d)
            alpha = np.linalg.pinv(K + lamb) @ y_train
            # in predicted y formula
            k_xi_x = (1 + x_validation * x_train[np.newaxis, :]) ** d   # use this when polynomial kernel
            # k_xi_x = np.exp(-gamma*np.linalg.norm(x_validation - x_train[np.newaxis, :], 2))
            y_predicted = alpha @ k_xi_x.T
            error_validation += (y_predicted - y_validation).T @ (y_predicted- y_validation)
            # error_validation = error_validation[0][0]
        error_validation /= n
        print("error_validation: ", error_validation)
        error_validation_list.append(error_validation)
        lamb_list.append(lamb)
        d_list.append(d)

min_error = min(error_validation_list)
index_boostrap_sample_min_error = error_validation_list.index(min(error_validation_list))
lamb_best_poly = lamb_list[index_boostrap_sample_min_error]
d_best = d_list[index_boostrap_sample_min_error]
print("Best lamb: ", lamb_best_poly, ", Best d: ", d_best)

# lamb_best_poly = 0.48828125
d_best = 30
# plots the comparaison
# np.random.seed(1)
x_fine = np.array(list(np.arange(min(x),max(x), 0.01))  )
n = len(x_fine)
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
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3b_1_test.png")
plt.show()


# A3.c1
B = 300
n = 30
n_fine = len(x_fine)
# np.random.seed(0)
boostrap_predicted_poly_matrix = []

for j in range(B):
    index_boostrap_sample = np.random.choice(n,n)
    x_training = x[index_boostrap_sample]
    y_training = y[index_boostrap_sample]
    K = k_poly(x_training[:,np.newaxis],x_training[:,np.newaxis], d_best)
    alpha = np.linalg.solve((K + lamb_best_poly*np.eye(n, n)), y_training)
    y_predicted_boostrap_ploy = []
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

plt.plot(x_fine, y_fine_true, label = 'True Model')
plt.plot(x_fine, f_poly_predicted, label = 'Poly Kernel Prediction')
plt.plot(x, y,'bo', label ='Observed data')
plt.fill_between(x_fine, percent_5_list_poly, percent_95_list_poly, alpha=0.3, label="90% CI")
plt.ylim(-6, 6)
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3c_1_test.png")
plt.show()


#######################################################################################################################
# A3.b2

def k_rbf(x, z, gamma):
    return np.exp(-gamma*(x-z)*(x-z))

n = 30
# np.random.seed(0)
# x = np.random.rand(n)
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
        for i in range(n):
            x_train = np.append(x[0:i], x[i+1:n])
            y_train = np.append(y[0:i], y[i+1:n])
            x_validation = x[i]
            y_validation = y[i]
            K = k_rbf(x_train[:,np.newaxis],x_train[np.newaxis,:], gamma)
            alpha = np.linalg.pinv(K + lamb) @ y_train
            k_xi_x = np.exp(-gamma*(x_validation-x_train[np.newaxis,:])**2)
            error_validation += (k_xi_x@alpha - y_validation).T@(k_xi_x@alpha - y_validation)
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


gamma_best= 10.175399541327897
lamb_best_rbf= 9.313225746154785e-07
# np.random.seed(10)

x_fine = np.arange(min(x),max(x),0.001)
n = len(x_fine)
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
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3b_2.png")
plt.show()

# A3.c2
B = 300
n=30
n_fine = len(x_fine)
# np.random.seed(0)
boostrap_predicted_rbf_matrix = []
# user x, y from previous
for j in range(B):
    index_boostrap_sample = np.random.choice(n,n)
    x_training = x[index_boostrap_sample]
    y_training = y[index_boostrap_sample]
    K_rbf = k_rbf(x_training[:, np.newaxis], x_training[np.newaxis, :], gamma_best)
    alpha = np.linalg.solve((K_rbf + lamb_best_rbf * np.eye(n, n)), y_training)

    y_predicted_boostrap_rbf = []
    for xi in x_fine:
        y_predicted_boostrap_rbf.append(np.sum(alpha * np.exp(-gamma_best*(xi-x_training)**2)))
    boostrap_predicted_rbf_matrix.append(y_predicted_boostrap_rbf)
boostrap_predicted_rbf_matrix = np.array(boostrap_predicted_rbf_matrix)

percent_5_list_rbf = []
percent_95_list_rbf = []
for i in range(n_fine):
    sorted_xi_from_300_B_sample = np.sort(boostrap_predicted_rbf_matrix[:, i])
    x_percentile_5 = sorted_xi_from_300_B_sample[int(B * 0.05)]
    x_percentile_95 = sorted_xi_from_300_B_sample[int(B * 0.95)]
    percent_5_list_rbf.append(x_percentile_5)
    percent_95_list_rbf.append(x_percentile_95)


plt.plot(x_fine, y_fine_true, label = 'True Model')
plt.plot(x_fine, f_rbf_predicted, label = 'rbf Kernel Prediction')
plt.plot(x, y,'bo', label ='Observed data')
plt.fill_between(x_fine, percent_5_list_rbf, percent_95_list_rbf, alpha=0.3, label="90% CI")
plt.ylim(-6, 6)
plt.legend()
plt.savefig("/Users/yinruideng/Desktop/senior_spring/cse546/hw/hw3/latex/plots/A3c_2_test.png")
plt.show()
#######################################################################################################################


