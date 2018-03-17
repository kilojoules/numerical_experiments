from secondsolver import geturec, xf
import numpy as np
from fd import d1

TOL = 0.01

def ulinf(u, dx): return np.max(np.abs(u))
def dulinf(u, dx): return np.max(np.abs(d1(u, dx)))
def ul2(u, dx): return np.sum(u ** 2) / u.size
def dul2(u, dx): return np.sum(d1(u, dx) ** 2) / u.size
def ul1(u, dx): return np.sum(np.abs(u)) / u.size
def dul1(u, dx): return np.sum(np.abs(d1(u, dx))) / u.size

QOIFS = [ulinf, dulinf, ul2, dul2, ul1, dul1]
QOINames = [r"$L_\infty(u)$", r"$L_\infty(\nabla u)$", r"$L_2(u)$", r"$L_2(\nabla u)",
            r"L_1(u)", r"L_1(\nabla u)"]
#nus = np.array([1])
nus = np.array([1, .1, .05, .04, .03, .029, .028, .027, .026, .025, .024, .023, .022, .021, 0.02, 
                .019, .018, .017, .016, .015, .014, .013, .012, .011, .01, .009, .008, .007, .005])
#nus = np.array([1, .5, .3, .1, .05, .04, .03, 0.02, .015, .01, .005])
f = open('xofnu.csv', 'w')
for nu in nus:
    lastqoi = None
    largestx = 0
    nx = 110
    while True:
        x = np.linspace(0, xf, nx) ; dx = x[1] - x[0]
        u = geturec(nu=nu, x=x, convstrategy='4c', diffstrategy='5c', timestrategy='vanilla', evolution_time = 0.1, n_save_t=20)
        qoi = np.array([QOIFS[kk](u[:, -1], dx) for kk in range(len(QOIFS))])
        if lastqoi is not None:
            print('nu: ', nu, ',  nx: ', nx, ",  QOI: ", QOINames, qoi, ",  res: ", (qoi - lastqoi) / qoi)
            if not np.isnan(np.max(lastqoi)) and np.max(np.abs(qoi - lastqoi) / qoi ) < TOL: break
        lastqoi = qoi.copy()
        nx *= 2
    if nx > largestx: 
         largestx = nx
    f.write(str(nu) + ', ' + str(largestx) +', '+' '.join([str(s) for s in qoi]) + "\n")
f.close()
