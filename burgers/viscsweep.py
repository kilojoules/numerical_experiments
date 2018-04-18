import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from mysolver import geturec

if __name__=="__main__":
     record = {}
     nus = [1, .5, .1, .05, .01, .005, .001, .0005, .0001]
     for ET in [.02, .05, .1, .5, 1, 1.2, 1.4, 1.6, 1.8, 2]:
        record[ET] = []
        for nu in nus:
           nx = 25
           oldu = None
           while True:
              x = np.linspace(0, np.pi, nx + 1)
              u = geturec(x, nu=nu, evolution_time=ET, n_save_t=1)
              #u = geturec(x, nu=nu, evolution_time=ET, n_save_t=1)[:, -1]
              if oldu is not None:
                 qoi = np.sqrt(np.sum((u[0::2] - oldu)**2) / (nx/2))
                 print(ET, nu, nx, qoi)
                 if qoi < 0.005:
                    record[ET].append(nx)
                    break
              nx *= 2
              oldu = u
     for ET in record.keys(): plt.plot(nus, record[ET], label=ET, marker='x')
     plt.legend()
     plt.xlabel('# x points')
     plt.ylabel(r'$\nu$')
     plt.xscale('log')
     plt.yscale('log')
     plt.savefig('ETs.pdf')
        #fl = open('mine.dat', 'w')
    ##l.write(' '.join([str(s) for s in u[:, -1]]))
    #fl.close()
