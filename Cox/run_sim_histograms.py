import numpy as np
import matplotlib.pyplot as plt
from required_functions import cox_model_cop, gen_model
import numpy.random as rnd
import time
from joblib import Parallel, delayed
import pandas as pd


def set_cov(corr, l, p):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma


def experiment(counter, GM, cox, vals, n):
    T, C, X = GM.gen(n)
    cox.fit(T, C, X)
    tic = time.time()
    cox.estimate_theta(vals)
    toc = time.time()
    elapsed_time = (toc-tic)/60
    print('theta_est = ' + str(cox.theta_est)+', process = '+str(counter)+', elapsed time = ' + str(elapsed_time))
    return cox.theta_est

def run_experiment(zeta, theta0):
    alphas = np.exp(np.linspace(np.log(4.0), np.log(1.0), 100))
    #sample size
    n = 500
    #number of covariates
    p =  int(zeta * n)
    #define the true parameters of the cumulative hazard
    phi0 = - np.log(2)
    rho0 = 2.0
    model = 'log-logistic'
    #define the interval in which the censoring is uniform
    tau1 = 1.0
    tau2 = 2.0
    #parameters covariance matrix spectrum
    A0 = set_cov(0.5, 7, p)
    #simulate the model parameters from the prior
    beta0 = rnd.normal(size = p)
    beta0 = theta0 * beta0 / np.sqrt(beta0 @ A0 @ beta0)
    #data generating process
    GM = gen_model(A0, beta0, phi0, rho0, tau1, tau2, model) 
    #fit frailty
    cox = cox_model_cop(p, alphas)
    vals = np.arange(0.01, 2.0, 0.01)
    m = 500
    thetas = np.zeros(m)
    for j in range(m):
        thetas[j] = experiment(j, GM, cox, vals, n)

    fmt = 'zeta_' + str(zeta)+'theta0_' + str(theta0)

    plt.figure()
    plt.hist(thetas, density = True, bins = np.arange(0.0, 2.0, 0.05))
    plt.axvline(x = theta0, color = 'red')
    plt.savefig('cox_hist' + fmt + '.png')

    data = {
        'theta' : thetas
    }

    df = pd.DataFrame(data)
    fmt ='_zeta'+"{:.2f}".format(zeta)+'_theta'+"{:.2f}".format(theta0)
    df.to_csv('data/sim_thetas'+fmt+'.csv', index = False)

    return 

theta0 = 1.0
for zeta in [0.7]:
    run_experiment(zeta, theta0)
