import numpy as np
from required_functions import rs_logit
import time
import pandas as pd 

#zeta 
zeta = 0.3
#theta0 
theta0 = 1.0
#theta values 
values = np.arange(5.0, 0.05, -0.05)
#data container
metrics = np.empty((len(values), 7))
#create the rs logit model object 
m = 10000
logit_rs = rs_logit(zeta, theta0, 0, m)
# loop over the values of lambda
for l in range(len(values)):
    alpha = values[l]
    #solve rs eqs
    logit_rs.solve(alpha)
    #loo_bs
    cv_bs_loo = logit_rs.cv_bs()
    cv_loss = logit_rs.cv_loss()
    res = np.array([alpha, logit_rs.w, logit_rs.v, logit_rs.tau, cv_bs_loo, cv_loss, logit_rs.corr], float)
    print(res)
    metrics[l,:] = res
df = pd.DataFrame(metrics, columns=['alpha', 'w', 'v', 'tau', 'cv_bs_loo', 'cv_loss', 'corr'])
df.to_csv('data/rs_zeta'+"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)+'.csv', index = False)

