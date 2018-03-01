import numpy as np
import matplotlib.pyplot as plt
from solver import geturec, xf

def maxgrad(u, dx): return np.max(np.abs(np.gradient(u[:-2, -1], dx)))

nus = np.array([1, .5, .3, .1, .05, 0.02, .01, .005])
nxs  = [25 * 2 ** ii for ii in range(10)]

flows = {}
books = {}
maxx = 0
for nu in nus:
    flows[nu] = {}
    books[nu] = {}
    lastx = None
    nx = 25
    ii = 1
    while True:
        print 'nx ', nx, '   nu ', nu
        x = np.linspace(0., xf, nx)
        dx = x[1] - x[0]
        # Re = u * dx / nu
        flows[nu][nx] = geturec(nu=nu, x=x, ub=1)
        books[nu][nx] = np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
        if lastx:
             res = np.abs(np.max(np.abs(np.gradient(flows[nu][lastx][:, -1], lastdx))) - np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))) / np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
             if res < 0.025:
                 break
        if lastx: print 'res ', res
        lastx = nx
        lastdx = dx
        nx = int(nx * 1.2)
        ii += 1
        if ii > maxx: maxx = ii
#maxx = max([len(flows[nu].keys()) for nu in nus])
colors = plt.cm.plasma(np.linspace(0, 1, maxx))
nx = 25
for ii in range(maxx):
    x = np.linspace(0., xf, nx)
    dx = x[1] - x[0]
    thesenus = np.array([nu for nu in nus if nx in flows[nu]])
    maxgs = np.array([maxgrad(flows[nu][nx], dx) for nu in thesenus])
    s = plt.scatter(thesenus, maxgs, marker='x', label="%i x-points"%nx, color=colors[ii])
    nx = int(nx * 1.2)

plt.xlabel(r"$\nu$")
plt.ylabel(r"max($|\nabla u|$)")
LGD = plt.legend(prop={'size':14}, ncol=3, bbox_to_anchor=[1, -.1])
plt.xscale('log')
plt.yscale('log')
plt.xlim(min(nus) * .8, max(nus) * 1.1)
plt.savefig('convergence.pdf', bbox_inches='tight', bbox_extra_artists=[LGD])
plt.clf()

xofnu = []
for nu in nus: 
    xofnu.append(max(flows[nu].keys()))
xofnu = np.array(xofnu)
plt.plot(nus, xofnu)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$')
plt.ylabel('Number of x-points needed for 5\% convergence')
plt.savefig('./xofnu.pdf')
plt.clf()
