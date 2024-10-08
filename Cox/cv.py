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
n = 400
#number of covariates
p =  int(zeta * n)
#true signal strength
theta0 = 1.0
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
cox.compute_cv_loss(verbose = True)


plt.figure()
plt.plot(alphas, cox.cv_loss, 'r-x', label = 'Vervweij - Van Houwelingen loo cross validation')
plt.plot(alphas, cox.approx_loo_loss, 'k-o', label = 'stat-mech approximation')
plt.legend()
plt.savefig('cox_loo_cv' + fmt + '.png')
plt.show()