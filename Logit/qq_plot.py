import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from required_functions import logit_model
import time
from joblib import Parallel, delayed
import pandas as pd 
from scipy.special import erf 

def Q(x):
    return 0.5 + 0.5*erf(x/np.sqrt(2))

#sample size
n = 1000
#covariate to sample size  ratio
zeta = 0.7
#number of covariates 
p = int(n * zeta)
#theta values 
values = np.arange(5.0, 0.05, -0.05)


def set_cov(corr, l):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma

theta0 = 1.0
fmt = '_zeta' +"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
beta0 = theta0 * beta0 / np.sqrt(beta0 @ beta0)
A0 = set_cov(0.5, 7)
inv_A0 = np.linalg.inv(A0)
#define fitting model
lg = logit_model(p, values)

data = lg.generate_random_instance(beta0, 0, A0, n)
lg.fit(data)
idx =  np.argmin(lg.w / lg.v)#np.argmin(lg.cv_loss)
beta = lg.betas[idx, :]
lg.estimate_theta()
print(lg.theta_est)
k = lg.w[idx] / lg.theta_est
print(k)
db_beta = beta /  k

t, x = data 
omega = np.zeros(p)
for j in range(p):
    Xj = x[:,j] 
    filt_col = (np.arange(0, p, 1) != j)
    Z = x[:, filt_col]
    gamma = np.linalg.inv(np.transpose(Z) @ Z) @ (np.transpose(Z) @ Xj)
    omega[j] = np.sqrt( np.mean((Xj - Z @ gamma)**2) / (1- zeta) )


std_res = lg.v[idx] / (omega * np.sqrt(p))
std_db_res = std_res / k
db_residuals = (db_beta - beta0) / std_db_res
residuals = (beta - beta0)/ std_res
# plt.figure()
# bins_edges = np.arange(-4, 4, 0.5)


# def gaussian(x, mu, sigma):
#     z = (x - mu) / sigma
#     return np.exp(- 0.5 * z **2) / ( sigma * np.sqrt(2 * np.pi))

# plt.title('Histogram residuals')
# # plt.hist(residuals, bins = bins_edges, color = 'green', density = True, alpha = 0.1)
# plt.plot(bins_edges, gaussian(bins_edges, 0, 1))
# plt.hist(db_residuals, bins = bins_edges, color = 'green', density = True)
# plt.axvline(x = 0.0, color = 'red')

ord_res = np.sort(residuals)
emp_quants = np.arange(1, len(ord_res) + 1, 1) / len(ord_res)
theo_quants = [Q(ord_res[j]) for j in range(len(ord_res))]

ord_res_db = np.sort(db_residuals)
theo_quants_db = [Q(ord_res_db[j]) for j in range(len(ord_res_db))]

plt.figure()
plt.title('QQ_plot')
# plt.plot(emp_quants, theo_quants, 'b.')
plt.plot(emp_quants, theo_quants_db, 'ko', label = 'de-biased')
plt.plot(emp_quants, emp_quants, 'r-')
plt.xlabel('empirical quantiles')
plt.ylabel('theoretical quantiles')
plt.savefig('figures/qq_plot'+fmt+'.png')

plt.figure()
plt.plot(beta0, db_beta, 'ko')
# plt.plot(beta0, beta, 'b.')
plt.plot(beta0, beta0, 'r.')
plt.xlabel(r'$\mathbf{\beta}_0$')
plt.ylabel(r'$\hat{\mathbf{\beta}}$')
plt.savefig('figures/scatter_plot'+fmt+'.png')




plt.show()
