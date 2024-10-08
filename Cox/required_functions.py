import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import math 
from scipy.special import lambertw
import scipy.optimize as opt 
from scipy.special import erf
import time
import warnings
import pandas as pd 
from numba import njit, vectorize
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

GH = np.loadtxt('GH.txt')
gp = np.sqrt(2)*GH[:,0]
gw = GH[:,1]/np.sqrt(np.pi)

@njit
def set_cov(corr, l, p):
    Sigma =  np.identity(p) 
    for k in range(1,l):
        Sigma = Sigma + (corr**k) * (np.diag(np.ones(p-k), k) + np.diag(np.ones(p-k), -k))
    return Sigma

@njit
def likelihood(z, c, a):
    return np.exp(c * z - a * np.exp(z))

@njit
def posterior(s,c,a):
    return (gw * likelihood(gp * s, c, a))/(gw @ likelihood(gp * s, c, a))

@vectorize
def posterior_expectation_elp(s, c, a):
    return posterior(s, c, a) @ np.exp(gp * s)

# vf = np.vectorize(f)
vl = np.vectorize(lambertw)

@njit 
def newton(c, x, beta0, alpha, tol=1.0e-8):
    err = 1.0
    its = 0
    lp = x @ beta0
    elp =  np.exp(lp) 
    H0 = na_est(c, 1, elp) 
    while(err >= tol):
        hess = H0 * elp
        S = (np.transpose(x) @ np.diag(hess + alpha) @ x) 
        phi = ((hess - c + alpha * lp) @ x)
        beta1 = beta0 - np.linalg.inv(S) @ phi
        lp = x @ beta1
        elp = np.exp(lp)
        H1 = na_est(c, 1, elp) 
        err = np.sqrt( sum((beta1-beta0)**2) + sum((H1-H0)**2))
        beta0 = beta1
        H0 = H1
        its = its+1 
    return beta0

@njit
def inv(y, x, z):
    err = 1.0
    while(err>1.0e-13):
        y = y - (1-z - np.mean(1.0/(1.0+x*y)))/(np.mean(x/(1.0+x*y)**2))
        err = abs(1-z - np.mean(1.0/(1.0+x*y)))
    return y

@njit
def breslow_est(c, event_type, elp):
    n = len(c)
    bh = np.zeros(n)
    R = sum(elp)
    if (c[0] == event_type):
        bh[0] = 1/R
    for i in range(1, n):
        R = R - elp[i-1]
        if (c[i]==event_type):
            bh[i] = 1/R
    return bh
@njit
def na_est(c, event_type, elp):
    n = len(c)
    ch = np.zeros(n)
    R = sum(elp)
    if (c[0] == event_type):
        ch[0] = 1/R
    else:
        ch[0] = 0 
    for i in range(1, n):
        ch[i] = ch[i-1]
        R = R - elp[i-1]
        if (c[i] == event_type):
            ch[i] = ch[i] + 1/R
    return ch


@njit
def cox_loss(c, lp):
    n = len(c)
    elp = np.exp(lp)
    R = sum(elp)
    loss = 0.
    if (c[0] == 1):
        loss = loss + np.log(R) - lp[0]  
    for i in range(1, n):
        R = R - elp[i-1]
        if (c[i] == 1):
            loss = loss + np.log(R) - lp[i]
    return loss 

@vectorize
def heaviside(x):
    y = 0
    if(x >= 0):
        y = 1.0
    return y

@vectorize
def phi(theta, alpha):
    res = gw @ np.exp(-np.exp(alpha + theta * gp))
    return res

@vectorize
def psi(theta, alpha):
    res = -  ( gw @  np.exp(alpha + theta * gp - np.exp(alpha + theta * gp)))
    return res

# @vectorize
@njit
def inv_phi(theta, y, tol = 1.0e-8):
    x = np.log(-np.log(y))
    err = 1.0
    while(err >= tol):
        update =  (phi(theta, x) - y) / psi(theta, x)
        x = x - update
        err = abs(update)#np.sqrt(sum(update**2))
    # print(y, x)
    return x

@njit
def hat_S(x, t):
    res = np.mean(heaviside(t-x))
    return res

class gen_model:
    def __init__(self, A, beta, phi, rho, t1, t2, model):
        self.p = len(beta)
        self.beta = beta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.A = A
        self.model = model

    def bh(self, t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self, t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def gen(self, n):
        X = rnd.multivariate_normal(mean=np.zeros(self.p), cov= self.A, size = n)
        lp = X @ self.beta
        u = rnd.random(size = n) 
        T0 = self.tau1 + u * (self.tau2-self.tau1)
        #sample the latent event times 
        u = rnd.random(size = n)
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(- np.log(u)) - lp - self.phi) / self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log( np.exp(- np.log(u) * np.exp(- lp)) - 1) - self.phi) / self.rho)
        #generate the observations 
        T = np.minimum(T1, T0)
        C = np.array(T1 < T0, int)
        return T, C, X
    
#class that contains function to fit the cox model 
class cox_model_cop:

    def __init__(self, p, alphas):
        self.p = p
        self.beta = np.zeros(self.p)
        self.alphas = alphas
        self.l = len(alphas)
        self.betas = np.zeros((self.l, self.p))
        self.w = np.zeros(self.l)
        self.v = np.zeros(self.l)
        self.tau = np.zeros(self.l)
        self.gamma = np.zeros(self.l)
        self.corr = np.zeros(self.l)
    
     
    def fit(self, t, c, x):
        #order the observations by their event times
        idx = np.argsort(t)
        self.t = np.array(t)[idx]
        self.c = np.array(c, int)[idx]
        self.x = x[[idx],:][0,:,:]
        self.n = len(t)
        self.dt = np.array([self.t[i+1] - self.t[i] for i in range(self.n-1)], float)
        self.Hs = np.zeros((self.l, self.n))
        self.H = np.zeros(self.n)
        self.beta = np.zeros(self.p)
        self.H_cens = na_est(self.c, 0.0, np.ones(self.n))
        self.h_cens = breslow_est(self.c, 0, np.ones(self.n))
        self.approx_loo_loss = np.zeros(self.l)
        for j in range(self.l):
            self.alpha = self.alphas[j]
            self.beta = newton(self.c, self.x, self.beta, self.alpha)
            self.betas[j,:] = self.beta
            self.H = na_est(self.c, 1, np.exp(self.x @ self.beta))
            self.h = breslow_est(self.c, 1, np.exp(self.x @ self.beta))
            self.w[j], self.v[j], self.tau[j], self.gamma[j], self.corr[j], self.approx_loo_loss[j] = cox_model_cop.compute_observables(self)
            self.Hs[j, :] = self.H
        return 
    
    def fit_frailty(self, theta):
        H0 = np.ones(self.n)
        err = 1.0
        its = 0
        while(err>=1.0e-8):
            elp = posterior_expectation_elp(theta, self.c, H0)
            H = na_est(self.c, 1, elp)
            err = np.sqrt(sum((H-H0)**2))
            H0 = H
            its = its + 1
        self.H_frailty = H0
        self.h_frailty = breslow_est(self.c, 1, elp)
        return 
    
    
    def compute_observables(self):
        self.zeta = self.p / self.n
        lp = self.x @ self.beta
        hess = self.H * np.exp(lp) + self.alpha
        score  = self.H * np.exp(lp) - self.c + self.alpha * lp
        tau = inv(self.zeta, hess, self.zeta)
        v = tau * np.sqrt(np.mean(score**2) / self.zeta)
        gamma = np.mean( lp ** 2)
        w = np.sqrt(max(gamma- (1-self.zeta) * v**2, 0))
        lp_loo = lp + tau * score
        corr = np.mean(self.t * lp_loo / w)
        a = sum([self.c[i] * np.log(self.h[i]) for i in range(self.n) if self.c[i] == 1]) / self.n
        approx_loo_loss = np.mean(self.H * np.exp(lp_loo) - self.c * lp_loo ) - a
        return w, v, tau, gamma, corr, approx_loo_loss
    
    def compute_w(self, theta0, idx_alpha):
        cox_model_cop.fit_frailty(self, theta0)
        w_rs = self.w[idx_alpha]
        v_rs = self.v[idx_alpha]
        tau_rs = self.tau[idx_alpha]
        alpha_rs = self.alphas[idx_alpha]
        weights = [(self.h_frailty[i] * np.exp(theta0 * gp)  + self.h_cens[i]) *np.exp( - self.H_frailty[i]*np.exp(theta0*gp)-self.H_cens[i]) for i in range(self.n)]
        lp = np.add.outer(w_rs * gp, v_rs * gp) / (1.0 + alpha_rs * tau_rs)
        mu = tau_rs/ (1.0 + alpha_rs * tau_rs)
        z =  [ lp + mu * self.c[i] for i in range(self.n)]
        xi = [ z[i]- np.array(vl(mu * self.H[i] * np.exp(z[i])), float) for i in range(self.n)]
        dg_xi = [self.H[i] * np.exp(xi[i]) - self.c[i] + alpha_rs * xi[i] for i in range(self.n)]
        dg_0 = [self.H_frailty[i] * np.exp(theta0 * gp) - self.c[i] for i in range(self.n)]
        w =  theta0 * tau_rs * sum([(gw * weights[i] * dg_0[i]) @ dg_xi[i] @ gw for i in range(self.n)])/self.zeta
        return w

    def parallel_create_list_values(self, values, idx_alpha):
        res = Parallel(n_jobs=12)(delayed(cox_model_cop.compute_w)(self, x, idx_alpha) for x in values)
        t_df = pd.DataFrame(res)
        rs_w_values = t_df.to_numpy()
        return rs_w_values 
    
    def estimate_theta(self, values, flag_exact_method = False):
        if(flag_exact_method):
            cox_model_cop.compute_cv_loss(self)
            idx_alpha = np.argmin(self.cv_loss)
        else:
            idx_alpha = np.argmin(self.approx_loo_loss)
        rs_w_values = cox_model_cop.parallel_create_list_values(self, values, idx_alpha)
        # print(rs_w_values)
        idx = np.argmin(np.abs(rs_w_values - self.w[idx_alpha]))
        self.theta_est = values[idx]
        k = self.w[idx_alpha] / self.theta_est
        self.beta_db = self.betas[idx_alpha, :]  / k 
        self.beta_best = self.betas[idx_alpha, :]
        self.v_db = self.v[idx_alpha] / k
        self.alpha_best = self.alphas[idx_alpha]
        self.H_best = self.Hs[idx_alpha, :]
        return
        
    def loo_loss(self, i, verb_flag):
        beta = np.zeros(self.p)
        loss = np.zeros(self.l)
        idx = np.arange(0, self.n, 1) != i
        x_i = self.x[idx, :]
        c_i = self.c[idx]
        tic = time.time()
        for j in range(self.l):
            alpha = self.alphas[j]
            beta = newton(c_i, x_i, beta, alpha)
            loss[j] = cox_loss(self.c, self.x @ beta) - cox_loss(c_i, x_i @ beta)
        toc = time.time()
        elapsed_time = (toc - tic) / 60
        if(verb_flag):
            print('observation = ' + str(i)+', time elapsed =' + str(elapsed_time))
        return loss
    
    def compute_cv_loss(self, verbose = False):
        res = Parallel(n_jobs=12)(delayed(cox_model_cop.loo_loss)(self, i, verb_flag = verbose) for i in range(self.n))
        t_df = pd.DataFrame(res)
        loo_values = np.stack(t_df.to_numpy())
        self.cv_loss = np.mean(loo_values, axis = 0)
        return 
    








class rs_cox_cop:

    def __init__(self, zeta, theta0, phi0, rho0, t1, t2, model, m):
        self.m = m
        self.w = 1.0e-3
        self.v = 1.0e-3
        self.tau = 1.0e-3
        GM = gauss_model(theta0, phi0, rho0, t1, t2, model)
        self.t, self.c, self.z0, self.q = GM.data_gen(m)
        self.H_true = GM.ch(self.t)
        self.H_null = na_est(self.c, 1, np.ones(self.m))
        self.H = self.H_null
        self.zeta = zeta
        self.theta0 = theta0
        self.var_mart_res_true = np.mean((self.H_true * np.exp(self.theta0 * self.z0) - self.c)**2)
    
    def solve(self, alpha):
        err = 1.0
        its = 0
        vareps = 0.5
        w0 = self.w
        v0 = self.v
        tau0 = self.tau
        H0 = self.H
        while (err>1.0e-8):
            z = (w0 * self.z0 + v0 * self.q + tau0 * self.c) / (1.0 + alpha * tau0)
            mu = tau0/ (1.0 + alpha * tau0)
            xi = z - np.array(vl(mu * H0 * np.exp(z)), float) 
            elp = np.exp(xi) 
            dg_xi = H0 * elp - self.c + alpha * xi
            dg_0 = self.H_true * np.exp(self.theta0 * self.z0) - self.c
            v1 = tau0 * np.sqrt (np.mean((dg_xi)**2) / self.zeta)   
            hess = H0 * elp + alpha   
            tau1 = inv(tau0, hess, self.zeta)
            w1 =  self.theta0 * tau0 * np.mean(dg_xi * dg_0) / self.zeta 
            H1 = na_est(self.c, 1, elp)
            v = vareps*v1 + (1-vareps)*v0
            w = vareps*w1 + (1-vareps)*w0
            tau = vareps*tau1 + (1-vareps)*tau0
            H = vareps*H1 + (1.0-vareps)*H0
            err = np.sqrt((v-v0)**2 + (w-w0)**2 + (tau-tau0)**2 + sum((H-H0)**2) )
            its = its + 1
            v0 = v
            w0 = w
            tau0 = tau
            H0 = H
            # print(alpha, w, v, err, its)
        self.w = w0
        self.v = v0
        self.tau = tau0
        self.H = H0
        self.xi = xi
        self.corr = np.mean(self.t * self.z0)
        return

    
class gauss_model:
    def __init__(self, theta, phi, rho, t1, t2, model):
        self.theta = theta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.model = model

    def bh(self,t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self,t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def data_gen(self, n):
        #generate the data
        Z0 = rnd.normal(size = n)
        Q = rnd.normal(size = n)
        u = rnd.random(size = n)
        lp = self.theta * Z0
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(-np.log(u))-lp-self.phi)/self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log(np.exp(-np.log(u)*np.exp(-lp))-1)-self.phi)/self.rho)
        if(self.model != 'weibull' and self.model != 'log-logistic'):
            raise TypeError("Only weibull and log-logistic are available at the moment") 
        u = rnd.random(size = n)
        T0 = (self.tau2 - self.tau1) * u + self.tau1
        T = np.minimum(T1,T0)
        C = np.array(T1<T0,int)
        #order the observations by their event times
        idx = np.argsort(T)
        T = np.array(T)[idx]
        C = np.array(C,int)[idx]
        Z0 = np.array(Z0)[idx]
        return T, C, Z0, Q