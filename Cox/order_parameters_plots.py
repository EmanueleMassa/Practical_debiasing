import matplotlib.pyplot as plt
import pandas as pd
zeta = 0.7
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
plt.plot(rs_df['alpha'],rs_df['corr'] ,'r-')
plt.errorbar(sim_df['alpha'],sim_df['corr_mean'],yerr =sim_df['corr_std'],fmt = 'ko', capsize = 3)

plt.ylabel(r"$\chi_n$")
plt.xlabel(r'$\alpha$')
plt.savefig('figures/correlation' + fmt +'.jpg')


# plt.figure()
# # plt.plot(rs_df['alpha'],rs_df['var_mart_res_true'] ,'g-')
# # plt.plot(rs_df['alpha'],zeta * rs_df['v']**2 /  rs_df['tau']**2 ,'b-')
# plt.plot(rs_df['alpha'], rs_df['tau'] / rs_df['v']**2 ,'r-')
# # plt.plot(rs_df['alpha'],zeta * rs_df['w'] / (rs_df['tau'] * theta0) ,'r-')

# # plt.errorbar(sim_df['alpha'],sim_df['var_mart_res_mean'],yerr =sim_df['var_mart_res_std'],fmt = 'ko', capsize = 3)
# # plt.plot(sim_df['alpha'],sim_df['var_mart_res_mean'])
# plt.xlabel(r'$\alpha$')


fig1 = plt.figure()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)

ax1.errorbar(sim_df['alpha'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'k.', capsize = 3)
ax1.plot(rs_df['alpha'],rs_df['w'],'r-')
ax1.set_ylabel(r'$w_n$')
ax1.set_xlim(left = 0.0, right = 3.0)
# ax1.set_xlabel(r'$\alpha$')

ax2.errorbar(sim_df['alpha'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'k.', capsize = 3)
ax2.plot(rs_df['alpha'],rs_df['v'],'r-')
ax2.set_ylabel(r'$v_n$')
ax2.set_xlim(left = 0.0, right = 3.0)
# ax2.set_xlabel(r'$\alpha$')

ax3.errorbar(sim_df['alpha'],sim_df['tau_mean'],yerr =sim_df['tau_std'],fmt = 'k.', capsize = 3)
ax3.plot(rs_df['alpha'],rs_df['tau'],'r-')
ax3.set_ylabel(r'$\tau_n$')
ax3.set_xlabel(r'$\alpha$')
ax3.set_xlim(left = 0.0, right = 3.0)

plt.savefig('figures/cox_estimates_order_parameters' + fmt + '.png')


fig2 = plt.figure()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)

ax1.errorbar(sim_df['alpha'],sim_df['gamma_mean'],yerr =sim_df['gamma_std'],fmt = 'ko', capsize = 3)
ax1.plot(rs_df['alpha'],rs_df['w']**2 + (rs_df['v']**2) * (1.0 - zeta),'r-')
ax1.set_ylabel(r'$\gamma_n$')
ax1.set_xlim(left = 0.0, right = 3.0)
# ax2.set_xlabel(r'$\alpha$')

ax2.plot(rs_df['alpha'],rs_df['corr'] ,'r-')
ax2.errorbar(sim_df['alpha'],sim_df['corr_mean'],yerr =sim_df['corr_std'],fmt = 'ko', capsize = 3)
ax2.set_ylabel(r"$\chi_n$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_xlim(left = 0.0, right = 3.0)
plt.savefig('figures/cox_estimates_metrics' + fmt + '.png')
plt.show()