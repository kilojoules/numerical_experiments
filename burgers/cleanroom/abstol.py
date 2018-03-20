from secondsolver import geturec, xf
import pandas as pd
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
dat = pd.read_csv('./xofnu01.csv')

f = open('ABSxofnu01.csv', 'w')
for jj in range(dat.nu.size):
    nu = dat.nu[jj]
    largestx = 0
    nx = 50
    trueqoi = np.array([float(s) for s in dat.qoi[jj].split(' ' )[1:]])
    while True:
        x = np.linspace(0, xf, nx) ; dx = x[1] - x[0]
        u = geturec(nu=nu, x=x, convstrategy='4c', diffstrategy='5c', timestrategy='rk', evolution_time=2, n_save_t=20, ubl=0, ubr=0)
        qoi = np.array([QOIFS[kk](u[:, -1], dx) for kk in range(len(QOIFS))])
        print('nu: ', nu, ',  nx: ', nx, ",  QOI: ", QOINames, qoi, ",  res: ", (qoi - trueqoi) / qoi)
        if not np.isnan(np.max(qoi)) and np.max(np.abs(qoi - trueqoi) / qoi ) < TOL: break
        lastqoi = qoi.copy()
        nx *= 1.2
    if nx > largestx: 
         largestx = nx
    f.write(str(nu) + ', ' + str(largestx) +', '+' '.join([str(s) for s in qoi]) + "\n")
f.close()
