from secondsolver import geturec, xf
import matplotlib.pyplot as plt
import numpy as np
from fd import d1

TOL = 0.002

def ulinf(u, dx): return np.max(np.abs(u))
def dulinf(u, dx): return np.max(np.abs(d1(u, dx)))
def ul2(u, dx): return np.sum(u ** 2) / u.size
def dul2(u, dx): return np.sum(d1(u, dx) ** 2) / u.size
def ul1(u, dx): return np.sum(np.abs(u)) / u.size
def dul1(u, dx): return np.sum(np.abs(d1(u, dx))) / u.size

qoiname = 'ul1'
QOIFS = [ulinf, dulinf, ul2, dul2, ul1, dul1]
QOINames = [r"$L_\infty(u)$", r"$L_\infty(\nabla u)$", r"$L_2(u)$", r"$L_2(\nabla u)",
            r"L_1(u)", r"L_1(\nabla u)"]
fbig, axbig = plt.subplots()
axbig.set_xscale('log')
axbig.set_ylabel(qoiname)
axbig.set_xlabel(r"$\nu$")
#nus = np.array([1])
#nus = np.array([1, .8, .6, .4, .3, .2, .1, .05, .04, .03, .02, .015, .013, .011, .01, .008, .006, .005, .004, .002, .001, .0008, .0006, .0004, .0002])
nus = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005])
fl = open('xofnu_%s_%f.csv'%(qoiname, TOL), 'w')
fl.write("nu,nx,qoi\n")
qs = []
xs = []
for nu in nus:
    lastqoi = None
    largestx = 0
    nx = 25
    xs.append([])
    qs.append([])
    while True:
        x = np.linspace(0, xf, nx) ; dx = x[1] - x[0]
        u = geturec(nu=nu, x=x, convstrategy='4c', diffstrategy='5c', timestrategy='rk', evolution_time=1, n_save_t=20, ubl=0, ubr=0)
        qoi = ul1(u[:, -1], dx)
        xs[-1].append(nx)
        qs[-1].append(qoi)
        #qoi = np.array([QOIFS[kk](u[:, -1], dx) for kk in range(len(QOIFS))])
        if lastqoi is not None:
            print('nu: ', nu, ',  nx: ', nx, ",  QOI: ", QOINames, qoi, ",  res: ", (qoi - lastqoi) / qoi)
            #if not np.isnan(np.max(lastqoi)) and np.max(np.log(np.abs(qoi / lastqoi)) / np.log(2) ) < TOL: break
            if not np.isnan(np.max(lastqoi)) and np.max(np.abs(qoi - lastqoi) / qoi ) < TOL: break
        lastqoi = qoi.copy()
        nx *= 2
    if nx > largestx: 
         largestx = nx
    f, ax = plt.subplots()
    ax.plot(u) ; f.savefig("%f_%s_ff01.pdf"%(nu, qoiname)) ; f.clf()
    axbig.scatter(np.ones(len(qs[-1]))*nu, qs[-1], marker='x', c=xs[-1], cmap=plt.cm.plasma) 
    fbig.savefig("conv_01_%s.pdf"%(qoiname)) 
    fl.write(str(nu) + ', ' + str(largestx) +', '+ str(qoi) + "\n")
    #fl.write(str(nu) + ', ' + str(largestx) +', '+' '.join([str(s) for s in qoi]) + "\n")
axbig.plot(nus, [qs[ii][-1] for ii in range(len(nus))])
fl.close()
fbig.savefig("conv_01_%s.pdf"%(qoiname))
plt.clf()
