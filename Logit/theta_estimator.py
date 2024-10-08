import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
# import scipy.optimize as opt 
from required_functions import logit_model
import time
from joblib import Parallel, delayed
import pandas as pd 

def experiment(counter, beta0, A0, lg, n):
    tic = time.time()
    #generate data
    data = lg.generate_random_instance(beta0, 0, A0, n)
    lg.fit(data)
    lg.estimate_theta()
    toc = time.time()
    print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
    return lg.theta_est


def set_cov(corr, l, p):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma



def run_sim(theta0, n, zeta, values, m):
    #number of covariates 
    p = int(n * zeta)
    #name
    fmt = '_zeta' +"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0) + '_n' + str(n)
    #simulate the model parameters from the prior
    beta0 = rnd.normal(size = p)
    beta0 = theta0 * beta0 / np.sqrt(beta0 @ beta0)
    A0 = set_cov(0.5, 7, p)
    #define fitting model
    lg = logit_model(p, values)
    tic = time.time()
    results = Parallel(n_jobs=12)(delayed(experiment)(counter, beta0, A0, lg, n) for counter in range(m))
    t_df = pd.DataFrame(results)
    thetas = t_df.to_numpy()
    toc = time.time()
    print('total elapsed time = ' + str((toc-tic)/60))

    t_df.to_csv('sim'+fmt+'.csv')

    plt.figure()
    bins_edges = np.arange(0.0, 2.0, 0.05)
    plt.hist(thetas, density = True, bins = bins_edges)
    plt.axvline(x = theta0, color = 'red')
    plt.savefig('histogram_theta'+fmt+'.png')
    return 





#sample size
n = 1000
#theta values 
theta0 = 1.0
#regularization path
values = np.exp(np.linspace(np.log(4.0), np.log(0.1), 100))
#repetitions
m = 1000

for zeta in [0.3, 0.7]:
    run_sim(theta0, n, zeta, values, m)