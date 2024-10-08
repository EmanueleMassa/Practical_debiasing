import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

zeta = 0.3
alpha = 1.0
theta0 = 1.0
fmt ='_zeta'+"{:.2f}".format(zeta)+'_theta'+"{:.2f}".format(theta0)
sim_df = pd.read_csv('data/sim_thetas'+fmt+'.csv')

plt.figure()
plt.hist(sim_df['theta'], density = True, bins = np.arange(0.0, 2.0, 0.05))
plt.axvline(x = theta0, color = 'red')
plt.savefig('figures/cox_hist' + fmt + '.png')
plt.show()