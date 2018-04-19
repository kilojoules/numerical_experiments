import numpy as np
from scipy.optimize import minimize as mini
import matplotlib.pyplot as plt
import numba as nb
from mysolver import geturec

if __name__=="__main__":
     NU = 0.05
     ET = 0.01
     TS = 'rk4'
     x = np.linspace(0,np.pi, 1001)
     trueu = geturec(x, nu=NU, evolution_time=ET, dt=1e-7, timestrategy=TS)[:, -1]
     y = []
     dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e1
     for dt in dts:
        print(dt)
        u = geturec(x, evolution_time=ET, dt=dt, nu=NU, n_save_t=1, timestrategy=TS)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     z = mini(lambda x: 1e15 * (x*dts[-1] - y[-1]) ** 2, [1]).x
     plt.plot(dts, dts * z, c='k', lw=3)
     a = mini(lambda x: 1e25 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [10]).x
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--')
     b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [20]).x
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--')[0]
     l.set_dashes([4, 4, 4, 2, 2, 2])
     c = mini(lambda x: 1e40 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [35]).x
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.')
     plt.plot(dts, y, c='r', marker='*')
     plt.yscale('log')
     plt.xscale('log')
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
