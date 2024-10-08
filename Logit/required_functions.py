import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd 
from numba import njit, vectorize
from joblib import Parallel, delayed
import pandas as pd
import time

GH = np.loadtxt('GH.txt')
gp = np.sqrt(2)*GH[:,0]
gw = GH[:,1]/np.sqrt(np.pi)

@njit
def f(theta):
    return gw @ (np.tanh(theta * gp ) * gp)

@njit
def df(theta):
    return gw @ ( gp * gp * (1.0 - np.tanh(theta * gp)**2))

@vectorize
def s_in(x, a):
    y = 0.
    if(x<(1+a) and x>(1-a)):
        y = x/(1+a)
    if(x<=(1-a)):
        y = x-a
    if(x>=(1+a)):
        y = x+a

    return y

# vs_in = np.vectorize(s_in)

@njit
def inv(y, x, zeta):
    err = 1.0
    while(err>1.0e-13):
        y = y - (1 - zeta - np.mean(1.0 / (1.0+x*y)))/np.mean(x/(1.0+x*y)**2)
        err = abs(1 - zeta - np.mean(1.0 / (1.0+x*y)))
    return y


@njit
def g(x, t):
    return np.log(np.cosh(x)) - t*x

@njit
def dg(x, t):
    return np.tanh(x) - t

@njit
def ddg(x):
    return 1- np.tanh(x)**2

@njit
def  prox_g(x, t, tau):
    err = 1.0
    z0 = s_in(x + tau * t, tau) 
    its = 0
    w = np.tanh(z0)
    while(err>1.0e-13):
        delta = -( z0 - x + tau * (w-t))/(1 + tau * (1 - w * w))
        z = z0 + delta
        w = np.tanh(z)
        err = np.max(np.abs(z - x + tau*(w-t)))
        its = its + 1
        z0 = z
    return z

@njit
def newton(beta, t, x, alpha, tol = 1.0e-8):
    err = 1.0
    its = 0
    lp = x @ beta 
    while(err >= tol):
        hess = np.diag(ddg(lp)) + alpha * np.identity(len(t))
        jac = np.transpose(x) @ hess @ x
        grad = (dg(lp, t) + alpha * lp )@ x
        beta_new = beta - np.linalg.inv(jac) @ grad
        lp = x @ beta_new 
        err = np.sqrt( sum((beta_new-beta)**2) )
        beta = beta_new
        its = its + 1 
        if(its > 1000):
            # print('for the love of god')
            break
    return beta
        
@njit
def bs_score(lp):
    res = np.mean( 0.25 * (1.0 - np.tanh(lp))**2)
    return res

@njit
def logit_loss(t, lp):
    loss = np.mean( np.log(2 * np.cosh(lp)) - lp * t)
    return loss

@njit
def compute_theta(x, theta_in, max_its = 300):
    theta = theta_in
    err = 1.0
    its = 0 
    flag = True
    update = 0.0
    while(err >= 1.0e-8 and flag):
        update = - (f(theta) - x) / df(theta)
        err = abs(update)
        theta = theta + update
        if(its >= max_its):
            flag = False
        its = its + 1
    if(flag == False):
        theta = np.inf
    return theta



class rs_logit:

    def __init__(self, zeta, theta0, phi0, m):
        self.m = m
        self.theta0 = theta0
        self.phi0 = phi0
        self.zeta = zeta
        self.t, self.z0, self.q =  rs_logit.generate_pop(self)
        self.w = 1.0e-3
        self.v = 1.0e-3
        self.tau = 1.0e-3
        self.xi = np.zeros(self.m)

    def generate_pop(self):
        z0 = rnd.normal(size = self.m)
        q = rnd.normal(size = self.m)
        u = rnd.random(size = self.m)
        lp = self.theta0 * z0
        prob = 0.5 * np.exp(lp) / np.cosh(lp)
        t = np.array(2 * np.array(u < prob, int) - np.ones(self.m), int)
        return t, z0, q
        
    def solve(self, alpha):
        self.alpha = alpha
        err = 1.0
        its = 0
        eta = 0.5
        w0 = self.w
        v0 = self.v
        tau0 = self.tau
        while (err > 1.0e-8):
            z = (w0 * self.z0 + v0 * self.q) / (1.0 + alpha * tau0)
            xi =  prox_g(z, self.t, tau0 / (1.0 + alpha * tau0))
            dg_xi = dg(xi, self.t) + alpha * xi
            dg_0 = dg(self.theta0 * self.z0, self.t) 
            v1 = np.sqrt(np.mean((tau0 * dg_xi)**2) / self.zeta)
            hess = ddg(xi) + alpha 
            tau1 = inv(tau0, hess, self.zeta) 
            w1 = self.theta0 * np.mean(tau0 * dg_xi * dg_0) / self.zeta 
            v = eta*v1 + (1-eta)*v0
            w = eta*w1 + (1-eta)*w0
            tau = eta*tau1 + (1-eta)*tau0
            err = np.sqrt((v-v0)**2 + (w-w0)**2 + (tau-tau0)**2)
            its = its + 1
            v0 = v
            w0 = w
            tau0 = tau
        self.w = w0
        self.v = v0
        self.tau = tau0
        self.xi = xi
        self.corr = self.theta0 * np.mean( 1.0 - np.tanh(self.theta0 * self.z0)**2 )
        return 
    
    def cv_bs(self):
        lp_loo = self.xi + self.tau * (np.tanh(self.xi) - self.t + self.alpha * self.xi ) 
        return bs_score(lp_loo)
    
    def cv_loss(self):
        lp_loo = self.xi + self.tau * (np.tanh(self.xi) - self.t + self.alpha * self.xi ) 
        return logit_loss(self.t, lp_loo)
    
    
#class that contains function to fit the logit model 
class logit_model:
    def __init__(self, p, alphas):
        self.p = p 
        self.alphas = alphas
        self.l = len(alphas)
        self.betas = np.zeros((self.l, self.p))
        
    def fit(self, data):
        self.t, self.x = data
        self.zeta = self.p / len(self.t) 
        self.beta = np.zeros(self.p)
        self.w = np.zeros(self.l)
        self.v = np.zeros(self.l)
        self.tau = np.zeros(self.l)
        self.gamma = np.zeros(self.l)
        self.cv_bs= np.zeros(self.l)
        self.cv_loss = np.zeros(self.l)
        self.corr = np.zeros(self.l)
        for j in range(self.l):
            self.alpha = self.alphas[j]
            self.beta = newton(self.beta, self.t, self.x, self.alpha)
            self.w[j], self.v[j], self.tau[j], self.gamma[j], self.cv_bs[j], self.cv_loss[j], self.corr[j] = logit_model.compute_observables(self)
            self.betas[j,:] = self.beta
        return
        

    def compute_observables(self):
        lp = self.x @ self.beta
        hess = -np.tanh(lp)**2 + 1  + self.alpha
        score  = np.tanh(lp) - self.t + self.alpha * lp
        tau = inv(self.zeta, hess, self.zeta)
        v = tau * np.sqrt(np.mean(score ** 2) / self.zeta)
        gamma = np.mean( lp ** 2 )
        # w = np.sqrt( max(gamma - (1.0 - self.zeta) * (v ** 2), 0))
        lp_loo = lp + tau * score
        w = np.sqrt( max(np.mean(lp_loo**2) - v ** 2, 0))
        cv_bs_loo = bs_score(lp_loo)
        cv_loss = logit_loss(self.t, lp_loo) 
        if(w > 0):
            corr = np.mean(self.t * lp_loo / w)
        else: 
            corr = 0.0
        return w, v, tau, gamma, cv_bs_loo, cv_loss, corr


        
    def generate_random_instance(self, beta0, phi0, A0, n):
        X = rnd.multivariate_normal(np.zeros(self.p), A0 , size = n)
        lp = X @ beta0 + phi0
        u = rnd.random(size = n)
        prob = 0.5 * (1.0 + np.tanh(lp))
        T = np.array(2 * np.array(u < prob, int) - np.ones(n), int)
        return T, X
    
    
    def estimate_theta(self):
        idx = np.argmax(self.cv_loss)
        theta_in = self.w[idx] * self.tau[idx] / (self.v[idx]**2)
        self.theta_est = compute_theta(self.corr[idx], theta_in)
        self.beta_best = self.betas[idx, :]
        k = self.w[idx] / self.theta_est
        self.beta_db = self.beta_best / k 
        self.v_db = self.v[idx] / k
        return 


    def compute_loo_loss(self, i):
        loo_loss = np.zeros(self.l)
        idx = (np.arange(0, self.n) != i)
        t_i, x_i = self.t[idx], self.x[idx, :]
        beta_i = np.zeros(self.p)
        tic = time.time()
        for j in range(self.l):
            alpha = self.alphas[j]
            beta_i = newton(beta_i, t_i, x_i, alpha)
            lp_i = self.x[i,:] @ beta_i
            loo_loss[j] = np.log(2 * np.cosh(lp_i)) - lp_i * self.t[i]
        toc = time.time()
        elapsed_time = (toc -tic) / 60
        print('process = ' + str(i)+', elapsed time = ' + str(elapsed_time))
        return loo_loss
    

    def compute_cv_loss(self):
        res = Parallel(n_jobs=12)(delayed(logit_model.compute_loo_loss)(self, i) for i in range(self.n))
        t_df = pd.DataFrame(res)
        loo_values = np.stack(t_df.to_numpy())
        self.cd_cv_loss = np.mean(loo_values, axis = 0)
        return 