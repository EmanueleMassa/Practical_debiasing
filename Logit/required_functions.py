import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd 
from numba import njit, vectorize

GH = np.loadtxt('GH.txt')
gp = np.sqrt(2)*GH[:,0]
gw = GH[:,1]/np.sqrt(np.pi)

# @njit
# def f1(x):
#     theta = x[0]
#     phi = x[1]
#     z = np.tanh(theta * gp + phi)
#     return gw @ z

# @njit
# def f2(x):
#     theta = x[0]
#     phi = x[1]
#     z = np.tanh(theta * gp + phi)
#     return gw @ ( (1.0 - z**2) * gp**2)

# @njit
# def df1(x):
#     theta = x[0]
#     phi = x[1]
#     z = np.tanh(theta * gp + phi)
#     return gw @ ( (1.0 - z**2) * gp**2)

# @njit
# def df2(x):
#     theta = x[0]
#     phi = x[1]
#     z = np.tanh(theta * gp + phi)
#     return gw @ ( (1.0 - z**2) * gp)

# @njit
# def df3(x):
#     theta = x[0]
#     phi = x[1]
#     z = np.tanh(theta * gp + phi)
#     return gw @ ( (1.0 - z**2))



# @njit
def f(x):
    theta = x[0]
    phi = x[1]
    z = np.tanh(theta * gp + phi)
    j1 = gw @ (gp * z)
    j2 = gw @ z
    res = np.array([j1, j2], float)
    return res

# @njit
def df(x):
    theta = x[0]
    phi = x[1]
    z = np.tanh(theta * gp + phi)
    j1 = gw @ ( (1.0 - z**2) * gp**2)
    j2 = gw @ ( (1.0 - z**2) * gp)
    j3 = gw @ ( (1.0 - z**2))
    res = np.array([[j1, j2], [j2, j3]], float)
    return res

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

# @njit
# def compute_theta(x, theta_in, max_its = 300):
#     theta = theta_in
#     err = 1.0
#     its = 0 
#     flag = True
#     update = 0.0
#     while(err >= 1.0e-8 and flag):
#         update = - (f(theta) - x) / df(theta)
#         err = abs(update)
#         theta = theta + update
#         if(its >= max_its):
#             flag = False
#         its = its + 1
#     if(flag == False):
#         theta = np.inf
#     return theta

# @njit
def compute_theta(x, y0, max_its = 300):
    y = y0
    f_val = f(y) - x
    err = 1.0
    its = 0 
    flag = True
    update = 0.0
    while(err >= 1.0e-8 and flag):
        jac = df(y)
        update = - np.linalg.inv(jac) @ f_val
        err = np.sqrt(sum(update**2))
        y = y + update
        f_val = f(y) - x
        if(its >= max_its):
            flag = False
        its = its + 1
    if(flag == False):
        y = np.array([np.inf, np.inf], float)
    # print(y)
    # print('err = '+str(err))
    return y


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
        self.varphi = 1.0e-3
        self.chi = np.zeros(self.m)

    def generate_pop(self):
        z0 = rnd.normal(size = self.m)
        q = rnd.normal(size = self.m)
        u = rnd.random(size = self.m)
        lp = self.theta0 * z0 + self.phi0
        prob = 0.5 * np.exp(lp) / np.cosh(lp)
        t = np.array(2 * np.array(u < prob, int) - np.ones(self.m), int)
        return t, z0, q
        
    
    def solve(self, alpha, verbose = False):
        err = 1.0
        its = 0
        eta = 0.5
        w0 = self.w
        v0 = self.v
        varphi0 = self.varphi
        tau0 = self.tau
        self.alpha = alpha
        while (err>1.0e-8):
            z =   (varphi0 + w0 * self.z0 + v0 * self.q ) / (1.0 + alpha * tau0)
            chi = prox_g(z, self.t, tau0 / (1.0 + alpha * tau0))
            dg_xi = dg(chi, self.t) + alpha * chi
            dg_0 = dg(self.theta0 * self.z0, self.t) 
            v1 = np.sqrt(np.mean((tau0 * dg_xi)**2) / self.zeta)
            hess = ddg(chi) + alpha 
            tau1 = inv(tau0, hess, self.zeta)
            w1 = self.theta0 * np.mean(tau0 * dg_xi * dg_0) / self.zeta 
            varphi1 =  np.mean(chi) 
            v = eta * v1 + (1 - eta) * v0
            w = eta * w1 + (1 - eta) * w0
            tau = eta * tau1 + (1 - eta) * tau0
            varphi = eta * varphi0 + (1 - eta) * varphi1
            err = np.sqrt((v - v0)**2 + (w - w0)**2 + (tau - tau0)**2  + (varphi - varphi0)**2)
            its = its + 1
            v0 = v
            w0 = w
            tau0 = tau
            varphi0 = varphi
        if(verbose):
            print(w, v, tau, varphi, err, its)
        self.w = w0
        self.v = v0
        self.tau = tau0
        self.varphi = varphi0
        self.chi = chi
        self.corr = self.theta0 * np.mean( 1.0 - np.tanh(self.phi0 + self.theta0 * self.z0)**2 )
        self.average = np.mean(np.tanh(self.phi0 + self.theta0 * self.z0))
        return 

    def cv_bs(self):
        lp_loo = self.chi - self.varphi + self.tau * (np.tanh(self.chi) - self.t + self.alpha * self.chi ) 
        return bs_score(lp_loo)
    
    def cv_loss(self):
        lp_loo = self.chi - self.varphi + self.tau * (np.tanh(self.chi) - self.t + self.alpha * self.chi )
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
        self.average = np.zeros(self.l)
        for j in range(self.l):
            self.alpha = self.alphas[j]
            self.beta = newton(self.beta, self.t, self.x, self.alpha)
            self.w[j], self.v[j], self.tau[j], self.gamma[j], self.cv_bs[j], self.cv_loss[j], self.corr[j], self.average[j] = logit_model.compute_observables(self)
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
        lp_loo = lp - self.beta[-1] + tau * score
        w = np.sqrt( max(np.mean(lp_loo**2) - v ** 2, 0))
        cv_bs_loo = bs_score(lp_loo)
        cv_loss = logit_loss(self.t, lp_loo) 
        if(w > 0):
            corr = np.mean(self.t * lp_loo / w)
        else: 
            corr = 0.0
        average = np.mean(self.t)
        return w, v, tau, gamma, cv_bs_loo, cv_loss, corr, average


        
    def generate_random_instance(self, beta0, phi0, A0, n):
        X = rnd.multivariate_normal(np.zeros(self.p-1), A0 , size = n)
        X0 = np.ones((n,1))
        X = np.hstack((X,X0))
        lp = X @ np.append(beta0, phi0)
        u = rnd.random(size = n)
        prob = 0.5 * (1.0 + np.tanh(lp))
        T = np.array(2 * np.array(u < prob, int) - np.ones(n), int)
        return T, X
    
    
    def estimate_theta(self):
        idx = np.argmax(self.cv_loss)
        x = np.array([self.corr[idx], self.average[idx]], float)
        self.theta_est, self.phi_est = compute_theta(x, np.zeros(2))
        self.beta_best = self.betas[idx, :-1]
        self.phi_best = self.betas[idx, -1]
        k = self.w[idx] / self.theta_est
        self.beta_db = self.beta_best / k 
        self.v_db = self.v[idx] / k
        return 
