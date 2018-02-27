import numpy as np
import matplotlib.pyplot as plt
from solver import geturec, xf
x = np.linspace(0., xf, 7000)

ub = 1

u = geturec(x=x, nu=.01, u0= ub + .8 * np.sin(x))

def pert(u0, u, pltnam):
    plt.plot(x, u[:, 0], label='Baseline Initial', lw=3)
    plt.plot(x, u0[:, 0], label='Perturbed Initial', ls='--', lw=3)
    plt.plot(x, u[:, -1], label='Baseline Final', lw=3)
    plt.plot(x, u0[:, -1], label='Perturbed Final', ls='--', lw=3)
    plt.xlabel('x')
    plt.ylabel('u')
    lgd = plt.legend(loc='lower center', bbox_to_anchor=[1.2, .5])
    plt.savefig(pltnam, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

# Perturbation 1 - high frequency, low amplitude
u0 = geturec(x=x, nu=.01, u0=ub + .2 * np.sin(30 * x) + .8 * np.sin(x))
pert(u0, u, 'hfla.pdf')

# Perturbation 2 - low frequency, low amplitude
u1 = geturec(x=x, nu=.01, u0=ub + .2 * np.sin(3 * x) + .8 * np.sin(x))
pert(u1, u, 'lfla.pdf')

# Perturbation 3 - low frequency, high amplitude
u2 = geturec(x=x, nu=.01, u0=ub + .8 * np.sin(3 * x) + .8 * np.sin(x))
pert(u2, u, 'lfha.pdf')

# Perturbation 4 - high frequency, high amplitude
u3 = geturec(x=x, nu=.01, u0=ub + .8 * np.sin(30 * x) + .8 * np.sin(x))
pert(u3, u, 'hfha.pdf')

# Do Different viscosities
nus = [.05, .01, .005]
us = []
u0s = []
for nu in nus:
    us.append(geturec(x=x, nu=nu, u0=1 + .8 * np.sin(x)))
    u0s.append(geturec(x=x, nu=nu, u0=1 + .5 * np.sin(7 * x) + .8 * np.sin(x)))

f, ax = plt.subplots(len(nus), figsize=(15, 15))
plt.subplots_adjust(bottom=-0.3)
lgds = []
for ii in range(len(nus)):
    ax[ii].plot(x, us[ii][:, 0], label="Baseline Initial", lw=4)
    ax[ii].plot(x, u0s[ii][:, 0], ls='--', label="Perturbed Initial", lw=4)
    ax[ii].plot(x, us[ii][:, -1], label="Baseline Final", lw=4)
    ax[ii].plot(x, u0s[ii][:, -1], ls='--', label="Perturbed Final", lw=4)
    ax[ii].set_xlabel('x', size=20)
    ax[ii].set_ylabel('u', size=20)
    ax[ii].set_title('nu = %.3f'%nus[ii], size=16)
    lgds.append(ax[ii].legend(loc='lower center', bbox_to_anchor=[1.1, 0]))
f.savefig('viscositySweep.pdf', bbox_extra_artists=lgds, bbox_inches='tight')
    #plt.clf()
