import numpy as np
import matplotlib.pyplot as plt
from solver import geturec, xf

def maxgrad(u, dx): return np.max(np.abs(np.gradient(u[:-2, -1], dx)))

nus = np.array([1, .5, .3, .1, .05, 0.02, .01, .005])
nxs  = [25 * 2 ** ii for ii in range(10)]

strategynames = {'4c': "4-term central", '3u': "3-term upwind", '2c': "2-term central", '2u':"2-term upwind"}

xofnurecord = {}

previouspoints = {}

for TOL in [0.1, .05, .025, .015]:
    xofnurecord[TOL] = {}
    for strategy in ['2u', '2c', '3u', '4c']:
        flows = {}
        books = {}
        maxx = 0
        for nu in nus:
            flows[nu] = {}
            books[nu] = {}
            lastx = None
            nx = 25
            if strategy in previouspoints: 
                if nu in previouspoints[strategy]:
                     nx = previouspoints[strategy][nu]
            ii = 1
            while True:
                print 'nx ', nx, '   nu ', nu
                x = np.linspace(0., xf, nx)
                dx = x[1] - x[0]
                # Re = u * dx / nu
                flows[nu][nx] = geturec(nu=nu, x=x, ub=1, strategy=strategy)
                books[nu][nx] = np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
                if lastx:
                     res = np.abs(np.max(np.abs(np.gradient(flows[nu][lastx][:, -1], lastdx))) - np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))) / np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
                     if res < TOL and not np.isnan(np.max(np.abs(np.gradient(flows[nu][lastx][:, -1], lastdx)))):
                         break
                if lastx: print 'res ', res
                lastx = nx
                lastdx = dx
                nx = int(nx * 1.2)
                ii += 1
                if ii > maxx: maxx = ii
        if strategy in previouspoints:
             if nu not in previouspoints[strategy]:
                 previouspoints[strategy][nu] = {}
        else: previouspoints[strategy] = {}
        previouspoints[strategy][nu] = lastx
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
        plt.savefig('convergence%s%.3f.pdf'%(strategy, TOL), bbox_inches='tight', bbox_extra_artists=[LGD])
        plt.clf()
        
        xofnu = []
        for nu in nus: 
            xofnu.append(max(flows[nu].keys()))
        xofnu = np.array(xofnu)
        xofnurecord[TOL][strategy] = xofnu.copy()

fig, ax = plt.subplots()
TOLs = xofnurecord.keys()
strats = strategynames.keys()
MS = ['x', '^', 'o', 'p']
colors = plt.cm.plasma(np.linspace(0, 1, len(strategynames.keys())))
for ii in range(len(TOLs)):
    for jj in range(len(strats)):
        ax.plot(nus, xofnurecord[TOLs[ii]][strats[jj]], c=colors[ii], marker=MS[jj], ms=4)
from matplotlib.lines import Line2D
custom_lines = []
nams = []
for jj in range(len(strats)):
    custom_lines.append(Line2D([0], [0], color='k', marker=MS[jj], lw=1, ms=4))
    nams.append(strategynames[strats[jj]])
for ii in range(len(TOLs)):
    custom_lines.append(Line2D([0], [0], color=colors[ii], lw=4))
    nams.append("Tolereance = %.3f"%TOLs[ii])
LGD = ax.legend(custom_lines, nams, bbox_to_anchor=[1, -.1], prop={'size':14})
ax.set_xlabel(r"$\nu$")
ax.set_ylabel("Required Number of x-points")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(min(nus) * .8, max(nus) * 1.1)
plt.savefig('sweepSummary.pdf', bbox_inches='tight', bbox_extra_artists=[LGD])
