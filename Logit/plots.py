import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
zeta = 0.3
theta0 = 1.0
fmt = '_zeta' +"{:.2f}".format(zeta)+'_theta0'+"{:.2f}".format(theta0)
rs_df = pd.read_csv('data/rs'+fmt+'.csv')
sim_df = pd.read_csv('data/sim'+fmt+'.csv')




plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['w'],'r-')
plt.ylabel(r'$w_n$')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/w' + fmt + '.jpg')

plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['v'],'r-')
plt.ylabel(r'$v_n$')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/v' + fmt + '.jpg')

plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['tau_mean'],yerr =sim_df['tau_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['tau'],'r-')
plt.ylabel(r'$\tau_n$')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/tau' + fmt + '.jpg')

plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['gamma_mean'],yerr =sim_df['gamma_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['w']**2 + (rs_df['v']**2) * (1.0 - zeta),'r-')
plt.ylabel(r'$\gamma_n$')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/gamma' + fmt + '.jpg')


plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['cv_bs_mean'],yerr =sim_df['cv_bs_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['cv_bs_loo'],'r-')
plt.ylabel(r'$CV BS $')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/cv_bs' + fmt + '.jpg')

plt.figure()
plt.errorbar(sim_df['alpha'],sim_df['cv_loss_mean'],yerr =sim_df['cv_loss_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alpha'],rs_df['cv_loss'],'r-')
plt.ylabel(r'$CV Loss $')
plt.xlabel(r'$\alpha$')
plt.savefig('figures/cv_loss' + fmt + '.jpg')



# plt.figure()
# plt.plot(rs_df['alpha'],rs_df['corr'] ,'r-')
# plt.errorbar(sim_df['alpha'],sim_df['corr_mean'],yerr =sim_df['corr_std'],fmt = 'ko', capsize = 3)
# # plt.plot(rs_df['alpha'], np.ones(len(rs_df['alpha']))*a, 'r-')
# plt.ylabel(r"$\chi_n$")
# plt.xlabel(r'$\alpha$')
# plt.savefig('correlation' + fmt +'.jpg')


fig1 = plt.figure()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)

ax1.errorbar(sim_df['alpha'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'k.', capsize = 3)
ax1.plot(rs_df['alpha'],rs_df['w'],'r-')
ax1.set_ylabel(r'$w_n$')
ax1.set_xlim(right = 3.0)
# ax1.set_xlabel(r'$\alpha$')

ax2.errorbar(sim_df['alpha'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'k.', capsize = 3)
ax2.plot(rs_df['alpha'],rs_df['v'],'r-')
ax2.set_ylabel(r'$v_n$')
ax2.set_xlim( left = 0.0, right = 3.0)
# ax2.set_xlabel(r'$\alpha$')

ax3.errorbar(sim_df['alpha'],sim_df['tau_mean'],yerr =sim_df['tau_std'],fmt = 'k.', capsize = 3)
ax3.plot(rs_df['alpha'],rs_df['tau'],'r-')
ax3.set_ylabel(r'$\tau_n$')
ax3.set_xlabel(r'$\alpha$')
ax3.set_xlim(right = 3.0)

plt.savefig('figures/estimates_oreder_parameters' + fmt + '.png')


# fig2 = plt.figure()
# ax1 = fig2.add_subplot(311)
# ax2 = fig2.add_subplot(312)
# ax3 = fig2.add_subplot(313)

# ax1.errorbar(sim_df['alpha'],sim_df['cv_loss_mean'],yerr =sim_df['cv_loss_std'],fmt = 'ko', capsize = 3)
# ax1.plot(rs_df['alpha'],rs_df['cv_loss'],'r-')
# ax1.set_ylabel(r'$\mathcal{L}_n$')
# ax1.set_xlim( left = 0.0, right = 3.0)

# # ax1.set_xlabel(r'$\alpha$')

# ax2.errorbar(sim_df['alpha'],sim_df['gamma_mean'],yerr =sim_df['gamma_std'],fmt = 'ko', capsize = 3)
# ax2.plot(rs_df['alpha'],rs_df['w']**2 + (rs_df['v']**2) * (1.0 - zeta),'r-')
# ax2.set_ylabel(r'$\gamma_n$')
# ax2.set_xlim( left = 0.0, right = 3.0)

# # ax2.set_xlabel(r'$\alpha$')

# ax3.plot(rs_df['alpha'],rs_df['corr'] ,'r-')
# ax3.errorbar(sim_df['alpha'],sim_df['corr_mean'],yerr =sim_df['corr_std'],fmt = 'ko', capsize = 3)
# ax3.set_ylabel(r"$\chi_n$")
# ax3.set_xlabel(r'$\alpha$')
# ax3.set_xlim( left = 0.0, right = 3.0)

fig2 = plt.figure()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)



ax1.errorbar(sim_df['alpha'],sim_df['gamma_mean'],yerr =sim_df['gamma_std'],fmt = 'ko', capsize = 3)
ax1.plot(rs_df['alpha'],rs_df['w']**2 + (rs_df['v']**2) * (1.0 - zeta),'r-')
ax1.set_ylabel(r'$\gamma_n$')
ax1.set_xlim( left = 0.0, right = 3.0)

# ax2.set_xlabel(r'$\alpha$')

ax2.plot(rs_df['alpha'],rs_df['corr'] ,'r-')
ax2.errorbar(sim_df['alpha'],sim_df['corr_mean'],yerr =sim_df['corr_std'],fmt = 'ko', capsize = 3)
ax2.set_ylabel(r"$\chi_n$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_xlim( left = 0.0, right = 3.0)

plt.savefig('figures/estimates_metrics' + fmt + '.png')
plt.show()
