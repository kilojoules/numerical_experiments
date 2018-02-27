import numpy as np
import matplotlib.pyplot as plt
from solver import geturec, xf

def maxgrad(u, dx): return np.max(np.abs(np.gradient(u[:-2, -1], dx)))

nus = np.array([1, .5, .3, .1, .05, 0.02, .01, .005])
nxs  = [25 * 2 ** ii for ii in range(10)]

flows = {}
books = {}
for nu in nus:
    flows[nu] = {}
    books[nu] = {}
    lastx = None
    nx = 25
    while True:
        print 'nx ', nx, '   nu ', nu
        x = np.linspace(0., xf, nx)
        dx = x[1] - x[0]
        # Re = u * dx / nu
        flows[nu][nx] = geturec(nu=nu, x=x, ub=1)
        books[nu][nx] = np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
        if lastx:
             res = np.abs(np.max(np.abs(np.gradient(flows[nu][lastx][:, -1], lastdx))) - np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))) / np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
             if res < 0.05:
                 break
        if lastx: print 'res ', res
        lastx = nx
        lastdx = dx
        nx *= 1.2
maxx = max([len(flows[nu].keys()) for nu in nus])
colors = plt.cm.viridis(np.linspace(0, 1, maxx))
for ii in range(maxx):
    nx = sorted(flows[nus[-1]].keys())[ii]
    x = np.linspace(0., xf, nx)
    dx = x[1] - x[0]
    plt.scatter([nu for nu in nus if nx in flows[nu]], np.array([maxgrad(flows[nu][nx], dx) for nu in nus if nx in flows[nu]]), marker='x', label="%i x-points"%nx, c=colors[ii])
    #plt.plot([nu for nu in nus if nx in flows[nu]], np.array([maxgrad(flows[nu][nx], dx) for nu in nus if nx in flows[nu]]), marker='x', label="%i x-points"%nx, c=colors[ii])

plt.xlabel(r"$\nu$")
plt.ylabel(r"max($\nabla |u|$)")
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
