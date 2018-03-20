import numpy as np
import matplotlib.pyplot as plt
from secondsolver import geturec, xf

QOI = "ulinf"
#def maxgrad(u, dx): return np.abs(u[:, -1]).sum() / float(u.shape[0])
#def maxgrad(u, dx): return np.sum(u[:, -1] ** 2) / float(u.shape[0])
if QOI == "dul2":
    def maxgrad(u, dx): return np.sum(np.gradient(u[:, -1], dx)**2) / float(u.shape[0])
elif QOI == "dul1":
    def maxgrad(u, dx): return np.sum(np.abs(np.gradient(u[:, -1], dx))) / float(u.shape[0])
elif QOI == "dulinf":
    def maxgrad(u, dx): return np.max(np.abs(np.gradient(u[:, -1], dx))) 
elif QOI == "ul2":
    def maxgrad(u, dx): return np.sum(u[:, -1]**2) / float(u.shape[0])
elif QOI == "ul1":
    def maxgrad(u, dx): return np.sum(np.abs(u[:, -1])) / float(u.shape[0])
elif QOI == "ulinf":
    def maxgrad(u, dx): return np.max(np.abs(u[:, -1]))
else: hey.you
ynams = {"dul2": r"$L_2(\nabla u)$", "dul1": r"$L_1(\nabla u)$", "dulinf": r"$L_\infty(\nabla u)$",
"ul2": r"$L_2(u)$", "ul1": r"$L_1(u)$", "ulinf": r"$L_\infty(u)$"}
          

nus = np.array([0.3])
#nus = np.array([1, .5, .3, .1, .05, .04, .03, 0.02, .015, .01, .005])
nxs  = [25 * 2 ** ii for ii in range(10)]

#strategynames = {'4c': "4-term central", '3u': "3-term upwind", '2c': "2-term central", '2u':"2-term upwind", 'rk': 'Runge-Kuta, 4-term central', '4c5d':"4-term central with 5-term central difussion"}
strategynames = {'4c': "4-term central", '2c': 'Two-term Central', '3c': "3-term Central", '3u': "3-term upwind", '2c': "2-term central", '2u':"2-term upwind", 'rk': 'Runge-Kuta, 4-term central', '5c':"5-term central"}
nxmult = 1.5
truthpoints = {
1: 100, .5: 200, .3: 250, .1: 350, .05: 550, .04: 700, .03: 750, .02: 850, .015: 1100,
.01: 1200, .005: 1600}
trueflows = {}
trueflowdxs = {}
for p in nus:
    print('generating truth for ',  p)
    v = p
    nx = int(truthpoints[p] * 6.)
    x = np.linspace(0, xf, nx)
    trueflows[v] = geturec(nu=v, x=x, convstrategy='4c', diffstrategy='5c')
    trueflowdxs[v] = x[1] - x[0]
print('truth generated')
hey
xofnurecord = {}
previouspoints = {}
#strats = ['4c5d', '4c']
convstrats = ['2c', '4c']
diffstrats = ['3c', '5c']
#strats = ['2u', '2c', '3u', '4c', '4c5d', 'rk']

TOLs = [0.01]
for TOL in TOLs:#, .005, .001]:
    xofnurecord[TOL] = {}
    for jj in range(len(convstrats) * len(diffstrats)):
    #for cs in convstrats:
    #  for ds in diffstrats:
    #for strategy in ['2u', '2c', '3u', '4c', 'rk']:
        div,rem = divmod(jj, 2) 
        cs = convstrats[div]
        ds =  diffstrats[rem]
        flows = {}
        books = {}
        maxx = 0
        for nu in nus:
            trudx = trueflowdxs[nu]
            flows[nu] = {}
            books[nu] = {}
            lastx = None
            nx = 25
            if cs in previouspoints: 
              if ds in previouspoints[cs]: 
                if nu in previouspoints[cs][ds]:
                     nx = previouspoints[strategy][nu]
            ii = 1
            while True:
                print('nx ', nx, '   nu ', nu, '   strategy ', cs, ds)
                x = np.linspace(0., xf, nx)
                dx = x[1] - x[0]
                # Re = u * dx / nu
                flows[nu][nx] = geturec(nu=nu, x=x, convstrategy=cs, diffstrategy=ds)
                books[nu][nx] = np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
                if lastx:
                     res = abs(maxgrad(flows[nu][nx], dx) - maxgrad(trueflows[nu], trudx)) / maxgrad(trueflows[nu], trudx)
                     #res = (maxgrad(flows[nu][nx], dx) - maxgrad(flows[nu][lastx], lastdx)) / maxgrad(flows[nu][nx], dx)
                     #res = np.abs(np.max(np.abs(np.gradient(flows[nu][lastx][:, -1], lastdx))) - np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))) / np.max(np.abs(np.gradient(flows[nu][nx][:, -1], dx)))
                     if res < TOL and not np.isnan(np.max(flows[nu][nx][:, -1])):
                         break
                if lastx: print('res ', res)
                lastx = nx
                lastdx = dx
                nx = int(nx * nxmult)
                ii += 1
                if ii > maxx: maxx = ii
        if cs in previouspoints:
           if ds in previouspoints[cs]:
              if nu not in previouspoints[cs][ds]:
                 previouspoints[cs][ds][nu] = {}
           else: previouspoints[cs][ds] = {}
        else: previouspoints[cs] = {}
        previouspoints[cs][ds][nu] = lastx
        #maxx = max([len(flows[nu].keys()) for nu in nus])
        colors = plt.cm.plasma(np.linspace(0, 1, maxx))
        nx = 25
        for ii in range(maxx):
            x = np.linspace(0., xf, nx)
            dx = x[1] - x[0]
            thesenus = np.array([nu for nu in nus if nx in flows[nu]])
            maxgs = np.array([maxgrad(flows[nu][nx], dx) for nu in thesenus])
            s = plt.scatter(thesenus, maxgs, marker='x', label="%i x-points"%nx, color=colors[ii])
            nx = int(nx * nxmult)
        plt.plot(nus, [maxgrad(trueflows[v], trueflowdxs[v]) for v in nus], c='k', label="Truth Data")
        plt.xlabel(r"$\nu$")
        plt.ylabel(ynams[QOI])
        #plt.ylabel(r"max($|\nabla u|$)")
        LGD = plt.legend(prop={'size':14}, ncol=3, bbox_to_anchor=[1, -.1])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(min(nus) * .8, max(nus) * 1.1)
        plt.savefig('convergence_%s_%s_%.3f.pdf'%(ds, dc, TOL), bbox_inches='tight', bbox_extra_artists=[LGD])
        plt.clf()
        
        xofnu = []
        for nu in nus: 
            xofnu.append(max(flows[nu].keys()))
        xofnu = np.array(xofnu)
        xofnurecord[TOL][cs][d] = xofnu.copy()

fig, ax = plt.subplots()
#TOLs = xofnurecord.keys()
MS = ['o', 'P', '*', '^', 'D', 'x']
colors = plt.cm.viridis(np.linspace(0, 1, len(TOLs)))
for ii in range(len(TOLs)):
    colors[ii][-1] = 0.5
    for jj in range(len(convstrats)*len(diffstrats)):
        div,mod = divmod(jj, 2) 
        ax.plot(nus, xofnurecord[TOLs[ii]][convstrats[div]][diffstrats[mod]], c=colors[ii], marker=MS[jj], ms=8)
from matplotlib.lines import Line2D
custom_lines = []
nams = []
for jj in range(len(convstrats)*len(diffstrats)):
    custom_lines.append(Line2D([0], [0], color='k', marker=MS[jj], lw=1, ms=10))
    div,mod = divmod(jj, 2) 
    nams.append(strategynames[convstrats[div]+" Convective, "+diffstrats[mod]+ "Diffusive"])
for ii in range(len(TOLs)):
    custom_lines.append(Line2D([0], [0], color=colors[ii], lw=4))
    nams.append("Tolereance = %.2f%s (QOI=%s)"%(TOLs[ii]*100, r'%', ynams[QOI]))
LGD = ax.legend(custom_lines, nams, bbox_to_anchor=[1, -.1], prop={'size':14})
ax.set_xlabel(r"$\nu$")
ax.set_ylabel("Required Number of x-points")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(min(nus) * .8, max(nus) * 1.1)
plt.savefig('sweepSummary.pdf', bbox_inches='tight', bbox_extra_artists=[LGD])
plt.clf()
