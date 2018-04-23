from completeSolver import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize as mini
nu = 0.7
nx = 800
x = np.linspace(0, np.pi, nx+1)
#ET=3.
#u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=100, timestrategy='fe', BCs='periodic', convstrategy='2c', diffstrategy='3c')
#colors = plt.cm.coolwarm(np.linspace(0, 1, u.shape[1]))
#for ii in range(u.shape[1]): plt.plot(u[:, ii], c=colors[ii])
#plt.savefig('uTime.pdf')

# Spatial Convergence - second order
ET=.002
nxs = [2 ** _ for _ in range(4, 9)]
nxs.reverse()
nxs = np.array(nxs)  * 2 ** 1
nx = nxs[0] * 4
print(nx)
x = np.linspace(0, np.pi, nx+1)
BC='dirchlet'
TS='fe'
trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy='2c', diffstrategy='3c')
truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy='2c', diffstrategy='3c', returndt=True)
y_2 = []
dxs = []
for ii, nx in enumerate(nxs):
    print(nx)
    x = np.linspace(0, np.pi, nx+1)
    dxs.append(x[1] - x[0])
    u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, dt=truedt, convstrategy='2c', diffstrategy='3c')
    #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
    y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
dxs = np.array(dxs)

def fitness(a): return 1e19 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
a = mini(fitness, 4).x

plt.plot(dxs, y_2, marker='*')
plt.plot(dxs, np.exp(a) * dxs ** 2, c='k')
plt.xscale('log')
plt.yscale('log')
plt.show()

