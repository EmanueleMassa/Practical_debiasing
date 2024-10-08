import numpy as np
import matplotlib.pyplot as plt
from required_functions import cox_model_cop, gen_model
import numpy.random as rnd
import time
from joblib import Parallel, delayed
import pandas as pd
from scipy.special import erf 

def Q(x):
    return 0.5 + 0.5 * erf(x/np.sqrt(2))



def set_cov(corr, l):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma


#covariate to sample size  ratio
zeta = 0.7
alphas = np.exp(np.linspace(np.log(4.0), np.log(0.5), 100))
#sample size
n = 1000
#number of covariates
p =  int(zeta * n)
#true signal strength
theta0 = 2.0
fmt = '_zeta' +"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)
#define the true parameters of the cumulative hazard
phi0 = - np.log(2)
rho0 = 2.0
model = 'log-logistic'
#define the interval in which the censoring is uniform
tau1 = 1.0
tau2 = 2.0
#parameters covariance matrix spectrum
A0 = set_cov(0.5, 7)
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
beta0 = theta0 * beta0 / np.sqrt(beta0 @ A0 @ beta0)
#data generating process
GM = gen_model(A0, beta0, phi0, rho0, tau1, tau2, model) 
T, C, X = GM.gen(n)
#fit frailty
cox = cox_model_cop(p, alphas)
cox.fit(T, C, X)
vals = np.arange(0.01, 2.0, 0.01)
tic = time.time()
cox.estimate_theta(vals)
toc = time.time()
elapsed_time = (toc-tic)/60
print(cox.theta_est, theta0 )
print('elapsed time = ' + str(elapsed_time))
print('best alpha = ' + str(cox.alpha_best))

beta = cox.beta_best
db_beta = cox.beta_db

sigma = np.zeros(p)
for j in range(p):
    Xj = X[:,j] 
    filt_col = (np.arange(0, p, 1) != j)
    Z = X[:, filt_col]
    gamma = np.linalg.inv(np.transpose(Z) @ Z) @ (np.transpose(Z) @ Xj)
    sigma[j] = np.sqrt( np.mean((Xj - Z @ gamma)**2) / (1- zeta) )


std_db_res = cox.v_db / (sigma * np.sqrt(p))
db_residuals = (db_beta - beta0) / std_db_res


ord_res_db = np.sort(db_residuals)
emp_quants = np.arange(1, len(ord_res_db) + 1, 1) / len(ord_res_db)
theo_quants_db = [Q(ord_res_db[j]) for j in range(len(ord_res_db))]

plt.figure()
plt.title('QQ_plot')
plt.plot(emp_quants, theo_quants_db, 'ko', label = 'de-biased')
plt.plot(emp_quants, emp_quants, 'r-')
plt.xlabel('empirical quantiles')
plt.ylabel('theoretical quantiles')
plt.savefig('figures/qq_plot'+fmt+'.png')

plt.figure()
plt.plot(beta0, beta, 'ko')
plt.plot(beta0, db_beta, 'b^')
plt.plot(beta0, beta0, 'r.')
plt.xlabel(r'$\mathbf{\beta}_0$')
plt.ylabel(r'$\hat{\mathbf{\beta}}$')
plt.savefig('figures/scatter_plot_beta'+fmt+'.png')

plt.figure()
cox.fit_frailty(cox.theta_est)
plt.plot(GM.ch(cox.t), cox.H, 'k-', ds = 'steps-post')
plt.plot(GM.ch(cox.t), cox.H_frailty, 'b-', ds = 'steps-post')
plt.plot(GM.ch(cox.t), GM.ch(cox.t), 'r-')
plt.xlabel(r'$\Lambda_0$')
plt.ylabel(r'$\hat{\mathbf{\Lambda}}_n$')
plt.savefig('figures/scatter_plot_H'+fmt+'.png')

plt.show()