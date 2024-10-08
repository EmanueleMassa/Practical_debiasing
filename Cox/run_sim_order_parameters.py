import numpy as np
import numpy.random as rnd
from required_functions import cox_model_cop, gen_model, rs_cox_cop, set_cov
import time
from joblib import Parallel, delayed
import pandas as pd 

def experiment(counter, cox, GM,  n):
    tic = time.time()
    #generate data
    t, c, x = GM.gen(n)
    cox.fit(t, c, x)
    toc = time.time()
    print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
    return cox.w, cox.v, cox.tau, cox.gamma, cox.corr

#sample size
n = 500
#covariate to sample size  ratio
zeta = 0.7
#number of covariates 
p = int(n * zeta)
#alpha values 
values = np.arange(5.0, 0.05, -0.05)
#theta value
theta0 = 1.0
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
#simulate a banded matrix with decaying correlation
A0 = set_cov(0.5, 7, p)
beta0 =  theta0 * beta0 / np.sqrt(beta0 @ A0 @ beta0)
#define the true parameters of the cumulative hazard
phi0 = - np.log(2)
rho0 = 2.0
model = 'log-logistic'
fmt = '_zeta'+"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)
#define the interval in which the censoring is uniform
tau1 = 1.0
tau2 = 2.0
#data generating process
GM = gen_model(A0, beta0, phi0, rho0, tau1, tau2, model) 
T, C, X = GM.gen(n)
#define fitting model
cox = cox_model_cop(p, values)


m = 50
tic = time.time()
results = Parallel(n_jobs=12)(delayed(experiment)(counter, cox, GM,  n) for counter in range(m))
t_df = pd.DataFrame(results)
w = np.stack(t_df.iloc[:, 0].to_numpy())
v = np.stack(t_df.iloc[:, 1].to_numpy())
tau = np.stack(t_df.iloc[:, 2].to_numpy())
gamma = np.stack(t_df.iloc[:, 3].to_numpy())
corr = np.stack(t_df.iloc[:, 4].to_numpy())
toc = time.time()
print('total elapsed time = ' + str((toc-tic)/60))


data = {
    'alpha' : values,
    'w_mean' : np.mean(w, axis = 0),
    'w_std' : np.std(w, axis = 0),
    'v_mean' : np.mean(v, axis = 0),
    'v_std' : np.std(v, axis = 0),
    'tau_mean' : np.mean(tau, axis = 0),
    'tau_std' : np.std(tau, axis = 0),
    'gamma_mean' : np.mean(gamma, axis = 0),
    'gamma_std' : np.std(gamma, axis = 0),
    'corr_mean' : np.mean(corr, axis = 0),
    'corr_std' : np.std(corr, axis = 0)
}

df = pd.DataFrame(data)
df.to_csv('data/sim' + fmt +'.csv', index = False)
    
metrics = np.empty((len(values), 6))
#create the rs cox model object 
m = 10000
cox_rs = rs_cox_cop(zeta, theta0, phi0, rho0, tau1, tau2, model, m)
# loop over the values of lambda
for l in range(len(values)):
    alpha = values[l]
    #solve rs eqs
    cox_rs.solve(alpha)
    res = np.array([alpha, cox_rs.w, cox_rs.v, cox_rs.tau, cox_rs.corr, cox_rs.var_mart_res_true], float)
    print(res)
    metrics[l,:] = res
df = pd.DataFrame(metrics, columns=['alpha', 'w', 'v', 'tau', 'corr', 'var_mart_res_true'])
df.to_csv('data/rs' +fmt + '.csv', index = False)







