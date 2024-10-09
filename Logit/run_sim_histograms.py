import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
# import scipy.optimize as opt 
from required_functions import logit_model
import time
from joblib import Parallel, delayed
import pandas as pd 

def experiment(counter, beta0, phi0, A0, lg, n):
    tic = time.time()
    #generate data
    data = lg.generate_random_instance(beta0, phi0, A0, n)
    lg.fit(data)
    lg.estimate_theta()
    toc = time.time()
    print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
    return lg.theta_est, lg.phi_est


def set_cov(corr, l):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma



#sample size
n = 1000
#covariate to sample size  ratio
zeta = 0.7
#number of covariates 
p = int(n * zeta)
#theta values 
values = np.arange(5.0, 0.05, -0.05)
#ripetitions
# m = 100

theta0 = 1.0
fmt = '_zeta' +"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
beta0 = theta0 * beta0 / np.sqrt(beta0 @ beta0)
phi0 = 0.5
A0 = set_cov(0.5, 7)
#define fitting model
lg = logit_model(p+1, values)

m = 500
tic = time.time()
results = Parallel(n_jobs=12)(delayed(experiment)(counter, beta0, phi0, A0, lg, n) for counter in range(m))
t_df = pd.DataFrame(results)
thetas = np.stack(t_df.iloc[:, 0].to_numpy())
phis = np.stack(t_df.iloc[:, 1].to_numpy())
toc = time.time()
print('total elapsed time = ' + str((toc-tic)/60))


plt.figure()
bins_edges = np.arange(0.0, 2.0, 0.05)
plt.hist(thetas, density = True, bins = bins_edges)
plt.axvline(x = theta0, color = 'red')
plt.savefig('histogram_theta'+fmt+'.png')


plt.figure()
bins_edges = np.arange(0.0, 2.0, 0.05)
plt.hist(phis, density = True, bins = bins_edges)
plt.axvline(x = phi0, color = 'red')
plt.savefig('histogram_phi'+fmt+'.png')

plt.show()




