import numpy as np
from scipy.optimize import minimize as mini
import matplotlib.pyplot as plt
import numba as nb
from mysolver import geturec

if __name__=="__main__":
     NU = 1e-3
     ET = 2e-2
     TS = 'rk2'
     x = np.linspace(0,np.pi, 2001)
     dts = np.linspace(1, 20, 5) * 1e-6
     maxdt = geturec(x, nu=NU, evolution_time=ET, timestrategy=TS, returndt=True)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=NU, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS)[:, -1]
     y = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=NU, n_save_t=1, timestrategy=TS)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        print('-->', dt, y[-1])
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     z = mini(lambda x: 1e15 * (x*dts[-1] - y[-1]) ** 2, [1]).x
     plt.plot(dts, dts * z, c='k', lw=3)
     a = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--')
     b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--')[0]
     l.set_dashes([1, 1])
     c = mini(lambda x: 1e50 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.')
     plt.plot(dts, y, c='r', marker='*')
     plt.yscale('log')
     plt.xscale('log')
     plt.ylabel(r'$\epsilon$')
     plt.xlabel(r'$t$')
     plt.savefig('time.pdf')
     plt.show()
     hey

     plt.xlabel(r'$\epsilon$')
     plt.ylabel(r'$t$')
     plt.xscale('log')
     plt.yscale('log')
     plt.savefig('time.pdf')
        #fl = open('mine.dat', 'w')
    ##l.write(' '.join([str(s) for s in u[:, -1]]))
    #fl.close()
