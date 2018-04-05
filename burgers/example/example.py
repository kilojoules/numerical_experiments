# Evolve the diffusion equation in time with Dirchlet BCs

# Load modules
import numpy as np
import matplotlib.pyplot as plt

# Domain size
XF = 1

# Viscosity
nu = 0.01

# Spatial Differentiation function, approximates d^u/dx^2
def diffusive_dudt(un, nu, dx, strategy='5c'):
    undiff = np.zeros(un.size, dtype=np.float128)

    # O(h^2)
    if strategy == '3c':
        undiff[2:-2] = nu * (un[3:-1] - 2 * un[2:-2] + un[1:-3]) / dx**2

    # O(h^4)
    elif strategy == '5c':
        undiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )
    else: raise(IOError("Invalid diffusive strategy")) ; quit()
    return undiff

# Velocity Evolution Function. Accepts initial and boundary conditions, returns time evolution history.
def geturec(x, nu=.05, evolution_time=1, u0=None, n_save_t=50, ubl=0., ubr=0., diffstrategy='5c', dt=None, returndt=False):

    dx = x[1] - x[0]

    # Prescribde cfl=0.05 and ftcs=0.02
    if dt is not None: pass
    else: dt = min(.05 * dx / 1., .02 / nu * dx ** 2)
    if returndt: return dt

    # Determine the interval, "divider", to record time with
    nt = int(evolution_time / dt)
    divider = int(nt / float(n_save_t))
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # The default initial condition is a half sine wave.
    u_initial = ubl + np.sin(x * np.pi)
    if u0 is not None: u_initial = u0
    u = u_initial
    u[0] = ubl
    u[-1] = ubr

    # insert ghost cells; extra cells on the left and right
    # for the edge cases of the finite difference scheme
    x = np.insert(x, 0, x[0]-dx)
    x = np.insert(x, -1, x[-1]+dx)
    u = np.insert(u, 0, ubl)
    u = np.insert(u, -1, ubr)

    # u_record holds all the snapshots. They are evenly spaced in time,
    # except the final and initial states
    u_record = np.zeros((x.size, int(nt / divider + 2)))

    # Evolve through time
    ii = 1
    u_record[:, 0] = u
    for _ in range(nt):
        un = u.copy()
        dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy)

        # forward euler time step
        u = un + dt * dudt

        # RK 4 time step
        #uhalfn = un + dt * dudt / 2.
        #duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy)
        #uhalfk2 = un + duhalfn_dt1 * dt / 2
        #duhalfk2_dt = diffusive_dudt(uhalfk2, nu, dx, strategy=diffstrategy)
        #ufull = un + duhalfk2_dt * dt
        #dufull_dt = diffusive_dudt(ufull, nu, dx, strategy=diffstrategy)
        #u = un + (dt / 6.) * (dudt + 2 * duhalfn_dt1 + 2 * duhalfk2_dt + dufull_dt)

        # Save every mth time step
        if _ % divider == 0:
            u_record[:, ii] = u.copy()
            ii += 1
    u_record[:, -1] = u
    return u_record[1:-1, :]

# define L-1 Norm
def ul1(u, dx): return np.sqrt(np.sum(u ** 2)) / u.size
#def ul1(u, dx): return np.sum(np.abs(u)) / u.size

# Now sweep through dxs to find convergence rate

# Define dxs to sweep
xrang = np.linspace(900, 1000, 6)

# this function accepts a differentiation key name and returns a list of dx and L-1 points
# the keynames are '3c' for 3 point centered differentiation and '5c' for 5 point centered differentiation
def errf(strat):

   # Lists to record dx and L-1 points
   ypoints = []
   dxs= []

   # Establish truth value with a more-resolved grid
   x = np.linspace(0, XF, 2000) ; dx = x[1] - x[0]

   # Record truth L-1 and dt associated with finest "truth" grid
   trueu = geturec(nu=nu, x=x, diffstrategy=strat, evolution_time=.05, n_save_t=20, ubl=0, ubr=0)
   truedt = geturec(nu=nu, x=x, diffstrategy=strat, evolution_time=.05, n_save_t=20, ubl=0, ubr=0, returndt=True) / 100.
   trueqoi = ul1(trueu[:, -1], dx)

   # Sweep dxs
   for nx in xrang:
       x = np.linspace(0, XF, nx) ; dx = x[1] - x[0]
       dxs.append(dx)

       # Run solver, hold dt fixed
       u = geturec(nu=nu, x=x, diffstrategy=strat, evolution_time=.05, n_save_t=20, ubl=0, ubr=0, dt=truedt)

       # record |L-1(dx) - L-1(truth)|
       qoi = ul1(u[:, -1], dx)
       ypoints.append(np.abs(trueqoi - qoi))

   return dxs, ypoints

# Plot results. The fourth order method should have a slope of 4 on the log-log plot.
from scipy.optimize import minimize as mini
strategy = '5c'
dxs, ypoints = errf(strategy)
def fit2(a): return 1000 * np.sum((a * np.array(dxs) ** 2 - ypoints) ** 2)
def fit4(a): return 1000 * np.sum((np.exp(a) * np.array(dxs) ** 4 - ypoints) ** 2)
a = mini(fit2, 500).x
b = mini(fit4, 14).x
plt.plot(dxs,  a * np.array(dxs)**2, c='k', label=r"$\nu^2$", ls='--')
plt.plot(dxs,  np.exp(b) * np.array(dxs)**4, c='k', label=r"$\nu^4$")
plt.plot(dxs, ypoints, label=r"Convergence", marker='x')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$\Delta X$")
plt.ylabel("$|L-L_{true}|$")
plt.title(r"$\nu=%f, strategy=%s$"%(nu, strategy))
plt.legend()
plt.savefig('%s.pdf'%strategy, bbox_inches='tight')
