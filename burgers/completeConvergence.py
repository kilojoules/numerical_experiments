from completeSolver import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize as mini
nu = 0.7
#nx = 800
#x = np.linspace(0, np.pi, nx+1)
#ET=1.
#u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=100, timestrategy='fe', BCs='periodic', convstrategy='2c', diffstrategy='3c')
#colors = plt.cm.coolwarm(np.linspace(0, 1, u.shape[1]))
#for ii in range(u.shape[1]): plt.plot(u[:, ii], c=colors[ii])
#plt.savefig('uTime.pdf')
#quit()

# Spatial Convergence - second order
if False:
    nu = 0.1
    ET=.004
    nxs = [2 ** _ for _ in range(4, 8)]
    nxs.reverse()
    nxs = np.array(nxs)  * 2 ** 1
    nx = nxs[0] * 4
    print(nx)
    x = np.linspace(0, np.pi, nx+1)[:-1]
    BC='periodic'
    TS='fe'
    cs = '2c'
    ds = '3c'
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)[:-1]
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1]))
        y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        #y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*')
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k')
    plt.plot(dxs, np.exp(a) * dxs, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon$')
    plt.savefig('2space.pdf')
    plt.clf()
                        

# Spatial Convergence - fourth order
if True:
    nu = 0.01
    ET=.0002
    nxs = [2 ** _ for _ in range(4, 9)]
    nxs.reverse()
    nxs = np.array(nxs)  * 2 ** 1
    nx = nxs[0] * 4
    print(nx)
    x = np.linspace(0, np.pi, nx+1)[:-1]
    BC='periodic'
    TS='fe'
    cs = '4c'
    ds = '5c'
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)[:-1]
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.abs(u[:, :] - trueu[0::2 **( ii + 2 ), :]))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*')
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k')
    plt.plot(dxs, np.exp(a) * dxs, c='k')
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('space4.pdf')
    plt.clf()


if False:
     NU = 4e-2
     ET = 6e-1
     TS = 'rk4'
     BC = 'periodic'
     x = np.linspace(0,np.pi, 11)
     dts = np.linspace(3, 4, 10) * 1e-3
     maxdt = geturec(x, nu=NU, evolution_time=ET, timestrategy=TS, returndt=True, BCs=BC)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=NU, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
     y = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=NU, n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
        y.append(np.max(np.abs(u - trueu)))
        #y.append(np.sqrt(np.sum((u - trueu) ** 2)))
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     z = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] - y[-1]) ** 2, [1]).x
     plt.plot(dts, dts * z, c='k', lw=3)
     a = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--')
     b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--')[0]
     l.set_dashes([1, 1])
     c = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
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

