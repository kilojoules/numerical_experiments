import numpy as np
from scipy.integrate import simps
import numba as nb
import matplotlib.pyplot as plt
from solver import geturec, xf, evolution_time
from fd import d1, d2

nx = 300
nu = .5
x = np.linspace(0, xf, nx)
dx = x[1] - x[0]
u_record = geturec(nu=nu, evolution_time = 1, x=x)
#ubar = np.zeros(u_record.shape[0]) #u_record.mean(1)
ubar = u_record.mean(1)
u_record = (u_record.T - ubar).T.copy()
print 'full model ran'

# SVD the covariance matrix
psi, D, phi = np.linalg.svd(u_record) # P D Q is the approximation

# choose # of modes to keep
MODES = 3

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
for i in range(MODES): plt.plot(Q[i, :], c=colors[i], label=i)
plt.legend(ncol=MODES, loc="upper center")
plt.ylabel(r'ai')
plt.xlabel('t')
plt.savefig('True_evolution_%i.pdf'%nx) ; plt.clf()

for ii in range(MODES): plt.plot(x, psi[:, ii], c=colors[ii])
plt.ylabel(r'$\psi_i$')
plt.xlabel('x')
plt.savefig('basis_%i.pdf'%nx) ; plt.clf()

@nb.jit
def genterms(MODES):
    bk1 = np.zeros(MODES)
    bk2 = np.zeros(MODES)
    Lik1 = np.zeros((MODES, MODES))
    Lik2 = np.zeros((MODES, MODES))
    Nijk = np.zeros((MODES, MODES, MODES))
    for kk in range(MODES):
        #bk1[kk] = simps(nu * psi[:, kk] *  np.gradient(np.gradient(ubar, dx), dx), dx=dx)
        #bk2[kk] = -1 * simps(ubar * psi[:, kk] *  np.gradient(ubar, dx), dx=dx)
        bk1[kk] += simps(nu * psi[:, kk] *  d2(ubar, dx))
        bk2[kk] += -1 * simps(ubar * psi[:, kk] *  d1(ubar, dx))
        for ii in range(MODES):
            Lik1[ii, kk] += nu * simps(psi[:, kk] * d2(psi[:, ii], dx))
            Lik2[ii, kk] += -1 * simps(ubar * psi[:, kk] *  d1(psi[:, ii], dx)
                                     + psi[:, ii] * d1(ubar, dx))
            #Lik1[ii, kk] = nu * simps(psi[:, kk] *  np.gradient(np.gradient(psi[:, ii], dx), dx), dx=dx)
            #Lik2[ii, kk] = -1 * simps(ubar * psi[:, kk] *  np.gradient(psi[:, ii], dx) 
                                     #+ psi[:, ii] * np.gradient(ubar, dx), dx=dx)
            for jj in range(MODES):
                Nijk[ii, jj, kk] += -1 * simps(psi[:, kk] * psi[:, ii] * d1(psi[:, jj], dx))
                #Nijk[ii, jj, kk] = -1 * simps(psi[:, kk] * psi[:, ii] * np.gradient(psi[:, jj], dx))
    return bk1, bk2, Lik1, Lik2, Nijk
b1, b2, L1, L2, N = genterms(MODES)
print 'generated terms'

#@nb.jit
def dqdt(_, a, ubar=ubar, nu=nu):
    #print '----> ', a
    dadt = np.zeros(a.shape[0])
    for kk in range(a.shape[0]):
        t1 = b1[kk] + b2[kk]
        for ii in range(a.shape[0]):
            t1 += nu * a[ii] * (L1[ii, kk] + L2[ii, kk])
            eddyvisc = 0.
            #eddyvisc =  (1e-1 + 0.05 * kk) * TERM1[ii, kk]
            for jj in range(a.shape[0]):
                t1 += a[jj] * a[ii] * (N[ii, jj, kk] + eddyvisc)
        dadt[kk] = t1.copy()
    return dadt

# record weights associated with first time :)
a0 = Q[:MODES, 0].copy()


########## Scipy ODE ROM implementation ###########
n_records = 0
t0 = 0
dt = 0.001
nt = int(evolution_time / dt)
n_save = nt / 2
divider = int(float(nt) / n_save)
u_red_record = np.zeros((nx, nt / divider + 1))
a_record = np.zeros((a0.size, nt / divider + 1))
from scipy.integrate import ode
r = ode(dqdt).set_integrator("dop853")
#r = ode(dqdt).set_integrator("lsoda", nsteps=100 * nt, ixpr=True, max_hnil=5)
#r = ode(dqdt).set_integrator("vode", nsteps=5000000, method="bdf")
#r = ode(dqdt)
r.set_initial_value(a0, t0)
ii = 0
while r.successful() and r.t < evolution_time:
    a = r.integrate(r.t+dt)
    ii += 1
    if ii % (divider) == 0:
        #print '----> ===', a, '====== ', r.t
        u_red_record[:, n_records] = np.dot(a.copy(), psi[:, :MODES].T).copy()
        a_record[:, n_records] = a.copy()
        n_records += 1
if not r.successful(): print "Solver Failure"
a_record[:, -1] = a.copy()
u_red_record[:, -1] = np.dot(a, psi[:, :MODES].T).copy()
####################################################
plt.plot(x, ubar + u_record[:, -1], c='k')
plt.plot(x, ubar + u_red_record[:, -1], c='r')
plt.title('%i Modes, nu=%f.4, nx=%i'%(MODES, nu, nx))
plt.show()

for ii in range(MODES):
    plt.plot(a_record[ii, :], label=ii)
plt.legend(ncol=MODES, loc="upper center")
plt.xlabel('Time')
plt.ylabel('ai')
plt.show()
