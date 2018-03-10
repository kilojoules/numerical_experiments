import numpy as np
from scipy.integrate import simps
import numba as nb
import matplotlib.pyplot as plt
from solver import geturec, xf, evolution_time
from fd import d1, d2

nx = 1000
nu = 0.1
x = np.linspace(0, xf, nx)
dx = x[1] - x[0]
u_record = geturec(nu=nu, evolution_time = 1, x=x)
print 'full model ran'

# SVD the covariance matrix
psi, D, phi = np.linalg.svd(u_record) # P D Q is the approximation

# choose # of modes to keep
MODES = 10

S = np.zeros((nx, u_record.shape[1]))
mm = min(nx, u_record.shape[1])
S[:mm, :mm] = np.diag(D)
plt.plot(np.sqrt(D)) ; plt.yscale('log') ; plt.savefig('spectrum.pdf') ; plt.clf()
assert(np.allclose(u_record, np.dot(psi, np.dot(S, phi)))) # check that a = P D Q
phi = phi.T
Q = np.dot(S[:, :MODES], phi[:, :MODES].T)

# Normalize the basis functions
for ss in range(phi.shape[1]):
    phi[:, ss] /= np.dot(phi[:, ss], phi[:, ss])
for ss in range(psi.shape[1]):
    psi[:, ss] /= np.dot(psi[:, ss], psi[:, ss])

# plot the modes
colors = plt.cm.coolwarm(np.arange(MODES) / (MODES-1.))
for i in range(MODES): plt.plot(Q[i, :], c=colors[i])
plt.savefig('Q_Modes.pdf') ; plt.clf()

@nb.jit
def genterms(MODES):
    t1 = np.zeros((MODES, MODES))
    t2 = np.zeros((MODES, MODES, MODES))
    for kk in range(MODES):
        for ii in range(MODES):
            t1[ii, kk] = simps(psi[:, kk] * d2(psi[:, ii], dx))
            for jj in range(MODES):
                t2[ii, jj, kk] = -1 * simps(psi[:, kk] * psi[:, ii] * d1(psi[:, jj], dx))
    return t1, t2
TERM1, TERM2 = genterms(MODES)
print 'generated terms'

#@nb.jit
def dqdt(_, a, nu=nu):
    #print '----> ', a
    dadt = np.zeros(a.shape[0])
    for kk in range(a.shape[0]):
        summ = 0
        for ii in range(a.shape[0]):
            summ += a[ii] * TERM1[ii, kk]
            eddyvisc = 0.
            #eddyvisc =  (1e-1 + 0.05 * kk) * TERM1[ii, kk]
            for jj in range(a.shape[0]):
                summ += a[ii] * a[jj] * TERM2[ii, jj, kk]
        dadt[kk] = summ.copy()
    return dadt

# record weights associated with first time :)
a0 = Q[:MODES, 0].copy()

########## Scipy ODE ROM implementation ###########
n_records = 0
t0 = 0
dt = 0.001
nt = int(evolution_time / dt) 
n_save = 200
divider = int(float(nt) / n_save)
u_red_record = np.zeros((nx, nt / divider + 1))
a_record = np.zeros((a0.size, nt / divider + 1))
#from scipy.integrate import ode
#r = ode(dqdt).set_integrator("dop853")
#r = ode(dqdt).set_integrator("lsoda", nsteps=100 * nt, ixpr=True, max_hnil=5)
#r = ode(dqdt).set_integrator("vode", nsteps=5000000, method="bdf")
#r = ode(dqdt)
#r.set_initial_value(a0, t0)
ii = 0
a = a0.copy()
for _ in range(nt):
#while r.successful() and r.t < evolution_time:
    update = + dqdt(_, a.copy()) * dt
    a += update
    #a = r.integrate(r.t+dt)
    ii += 1
    if ii % (divider) == 0:
        #print '----> ===', a, '====== ', r.t
        u_red_record[:, n_records] = np.dot(a.copy(), psi[:, :MODES].T).copy()
        a_record[:, n_records] = a.copy()
        n_records += 1
#if not r.successful(): print "Solver Failure"
a_record[:, -1] = a.copy()
u_red_record[:, -1] = np.dot(a, psi[:, :MODES].T).copy()
####################################################
plt.plot(x, u_record[:, -1], c='k')
plt.plot(x, u_red_record[:, -1], c='r')
plt.title('%i Modes'%MODES)
plt.show()
