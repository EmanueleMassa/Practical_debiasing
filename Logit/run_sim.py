import numpy as np
# import matplotlib.pyplot as plt
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
    toc = time.time()
    print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
    return lg.w, lg.v, lg.tau, lg.gamma, lg.cv_bs, lg.cv_loss, lg.corr


def set_cov(corr, l):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma

#sample size
n = 500
#covariate to sample size  ratio
zeta = 0.7
#number of covariates 
p = int(n * zeta)
#alpha 
# alpha = 0.1
#theta values 
values = np.arange(5.0, 0.05, -0.05)
#ripetitions
# m = 100

theta0 = 1.0
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
A0 = set_cov(0.5, 7)
beta0 =  theta0 * beta0 / np.sqrt(beta0 @ A0 @ beta0)
#define fitting model
lg = logit_model(p, values)

# data = lg.generate_random_instance(beta0, 0, A0, n)
# lg.fit(data)

# plt.figure()
# plt.plot(values, lg.cv_bs_loo, 'k.')

# plt.figure()
# plt.plot(values, lg.w *lg.tau / (lg.v **2), 'k.')

# plt.show()

# print(experiment(1, beta0, A0, lg, n))
m = 50
tic = time.time()
results = Parallel(n_jobs=12)(delayed(experiment)(counter, beta0, A0, lg, n) for counter in range(m))
t_df = pd.DataFrame(results)
w = np.stack(t_df.iloc[:, 0].to_numpy())
# print(np.shape(w))
v = np.stack(t_df.iloc[:, 1].to_numpy())
# print(np.shape(v))
tau = np.stack(t_df.iloc[:, 2].to_numpy())
# print(np.shape(tau))
gamma = np.stack(t_df.iloc[:, 3].to_numpy())
cv_bs = np.stack(t_df.iloc[:, 4].to_numpy())
cv_loss = np.stack(t_df.iloc[:, 5].to_numpy())
corr = np.stack(t_df.iloc[:, 6].to_numpy())
# print(np.shape(theta_est))
toc = time.time()
print('total elapsed time = ' + str((toc-tic)/60))

# print(np.shape(np.mean(w, axis = 0)))

data = {
    'alpha' : values,
    'w_mean' : np.mean(w, axis = 0),
    'w_std' : np.std(w, axis = 0),
    'v_mean' : np.mean(v, axis = 0),
    'v_std' : np.std(v, axis = 0),
    'tau_mean' : np.mean(tau, axis = 0),
    'tau_std' : np.std(tau, axis = 0),
    'cv_bs_mean' : np.mean(cv_bs, axis = 0),
    'cv_bs_std' : np.std(cv_bs, axis = 0),
    'gamma_mean' : np.mean(gamma, axis = 0),
    'gamma_std' : np.std(gamma, axis = 0),
    'cv_loss_mean' : np.mean(cv_loss, axis = 0),
    'cv_loss_std' : np.std(cv_loss, axis = 0),
    'corr_mean' : np.mean(corr, axis = 0),
    'corr_std' : np.std(corr, axis = 0)
}

df = pd.DataFrame(data)
df.to_csv('data/sim_zeta'+"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)+'.csv', index = False)
    








