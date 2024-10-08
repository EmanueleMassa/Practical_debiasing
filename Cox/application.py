import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

from sksurv.datasets import load_breast_cancer
from sksurv.preprocessing import OneHotEncoder

from sklearn import set_config
from sklearn.preprocessing import StandardScaler

from required_functions import cox_model_cop

X, Y = load_breast_cancer()
n = len(Y)
Xt = OneHotEncoder().fit_transform(X)
t = np.array([ Y[i][1] for i in range(n)], float)
c = np.array([Y[i][0] for i in range(n)], int)
x = Xt.to_numpy()
x = np.array(x, float)
# print(type(x), np.shape(x))
p = len(x[0, :])
alphas = np.exp(np.linspace(np.log(4.0), np.log(0.1), 100))
cox_m = cox_model_cop(p, alphas)
cox_m.fit(t, c, x)
cox_m.compute_cv_loss(verbose = True)

plt.figure()
plt.title('Elbow plot')
plt.plot(alphas, cox_m.betas)
plt.xlabel(r'$\alpha$')
plt.savefig('elbow_plot_application.png')

plt.figure()
plt.plot(alphas, cox_m.cv_loss, 'r-x', label = 'Vervweij - Van Houwelingen loo cross validation')
plt.plot(alphas, cox_m.approx_loo_loss, 'k-o', label = 'stat-mech approximation')
plt.legend()
plt.ylabel('CV loss')
plt.xlabel(r'$\alpha$')
plt.savefig('cox_cv_loss_applciation.png')

vals = np.arange(0.01, 2.0, 0.01)
tic = time.time()
cox_m.estimate_theta(vals)
toc = time.time()
elapsed_time = (toc-tic)/60
print('theta_est = ' + str(cox_m.theta_est) + ', elapsed time = '+ str(elapsed_time))

beta = cox_m.beta_best
db_beta = cox_m.beta_db

plt.figure()
plt.plot(db_beta, 'ko', label = 'select via cv')
plt.plot(beta, 'bo', label = 'de-biased')
plt.legend()
plt.savefig('plot_beta_application.png')

plt.figure()
cox_m.fit_frailty(cox_m.theta_est)
plt.plot(cox_m.t, cox_m.H_best, 'k-', ds = 'steps-post', label = 'select via cv')
plt.plot(cox_m.t, cox_m.H_frailty, 'b-',ds = 'steps-post', label = 'de-biased')
plt.xlabel(r'$t$')
plt.ylabel(r'$\hat{\Lambda}_n(t)$')
plt.legend()
plt.savefig('H_application.png')


plt.show()


