# Evolve the diffusion equation in time with Dirchlet BCs

# Load modules
import numpy as np
import matplotlib.pyplot as plt

# Domain size
XF = 1

# Viscosity
nu = 0.04

# Convective Differentiation function, approximates du/dx
def convective_dudt(un, dx, strategy='4c'):
    duconv = np.zeros(un.size, dtype=np.float128)
    if strategy=='2c': 
        duconv[1:-1] = -1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx)
    elif strategy == '4c':
        #duconv[1] = -1 * un[1] * (-10 * un[0] - 77 * un[1] + 150 * un[2] - 100 * un[3] + 50 * un[4] -15 * un[5] + 2 * un[6]) / 60 / dx
        #duconv[-2] = -1 * un[-2] * (10 * un[-1] + 77 * un[-2] - 150 * un[-3] + 100 * un[-4] - 50 * un[-5] + 15 * un[-6] - 2 * un[6]) / 60 / dx
        #duconv[1] = -1 * un[1] * (un[2] - un[0]) / (2 * dx)
        #duconv[-2] = -1 * un[-2] * (un[-1] - un[-3]) / (2 * dx) 
        duconv[1] = -1 * un[1] * ( - 25./12. * un[1] + 4 * un[2] - 3 * un[3] + 4./3. * un[4] - un[5]/4.) / dx
        duconv[-2] = -1 * un[-2] * ( - 25./12. * un[-2] + 4 * un[-3] - 3 * un[-4] + 4./3. * un[-5] - un[-6]/4.) / dx
        duconv[2:-2] = -1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx)
    return duconv

# Diffustive Differentiation function, approximates nu d^2 u /dx^2
def diffusive_dudt(un, nu, dx, strategy='5c'):
    dundiff = np.zeros(un.size, dtype=np.float128)

    # O(h^2)
    if strategy == '3c':
        dundiff[1:-1] = nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2

    # O(h^4)
    elif strategy == '5c':
        # http://web.media.mit.edu/~crtaylor/calculator.html
        #dundiff[1] = nu * (137 * un[0] - 147 * un[1] - 255 * un[2] + 470 * un[3] - 285 * un[4] + 93 * un[5] - 13 * un[6]) / 180 / dx**2
        #dundiff[-2] = nu * (137 * un[-1] - 147 * un[-2] - 255 * un[-3] + 470 * un[-4] - 285 * un[-5] + 93 * un[-6] - 13 * un[-7]) / (180 * dx**2)

        # second order
        #dundiff[1] = nu * (un[0] - 2 * un[1] + un[2]) / dx ** 2
        #dundiff[-2] = nu * ( un[-1] - 2 * un[-2] + un[-3]) / dx ** 2
        dundiff[1] = nu * (15./4. * un[1] - 77./6. * un[2] + 107./6. * un[3] - 13 * un[4] + (61./12.) * un[5] - 5./6. * un[6]) / dx ** 2
        dundiff[-2] = nu * (15./4. * un[-2] - 77./6. * un[-3] + 107./6. * un[-4] - 13 * un[-5] + (61./12.) * un[-6] - 5./6. * un[-7]) / dx ** 2
        dundiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )
    else: raise(IOError("Invalid diffusive strategy")) ; quit()
    return dundiff

# Velocity Evolution Function. Accepts initial and boundary conditions, returns time evolution history.
def geturec(x, nu=.05, evolution_time=1, u0=None, n_save_t=50, ubl=0., ubr=0., diffstrategy='5c', convstrategy='4c', dt=None, returndt=False):

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
    #x = np.insert(x, 0, x[0]-dx)
    #x = np.insert(x, -1, x[-1]+dx)
    #u = np.insert(u, 0, ubl)
    #u = np.insert(u, -1, ubr)

    # u_record holds all the snapshots. They are evenly spaced in time,
    # except the final and initial states
    u_record = np.zeros((x.size, int(nt / divider + 2)))

    # Evolve through time
    ii = 1
    u_record[:, 0] = u
    for _ in range(nt):
        un = u.copy()
        dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy) + convective_dudt(un, dx, strategy=convstrategy)

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
    return u_record
    #return u_record[1:-1, :]

# Now sweep through dxs to find convergence rate

# Define dxs to sweep
nsweep = 12
xrang = [3]
for ii in range(1, nsweep):
   xrang.append(xrang[-1] + 2 ** ii)
xrang = xrang[4:]
xrang.reverse()
print (xrang)

# this function accepts a differentiation key name and returns a list of dx and L-1 points
# the keynames are '3c' for 3 point centered differentiation and '5c' for 5 point centered differentiation
def errf(diffstrategy, convstrategy):

   # Lists to record dx and L-1 points
   ypoints = []
   dxs= []

   # Establish truth value with a more-resolved grid
   print('establishing truth with ', xrang[0] + 2 ** (nsweep))
   x = np.linspace(0, XF, xrang[0] + 2 ** (nsweep)) ; dx = x[1] - x[0]

   # Record truth L-1 and dt associated with finest "truth" grid
   trueu = geturec(nu=nu, x=x, diffstrategy=diffstrategy, convstrategy=convstrategy, evolution_time=.02, n_save_t=20, ubl=0, ubr=0)
   truedt = geturec(nu=nu, x=x, diffstrategy=diffstrategy, convstrategy=convstrategy, evolution_time=.02, n_save_t=20, ubl=0, ubr=0, returndt=True)
   #trueqoi = ul1(trueu[:, -1], dx)
   us = [trueu[:, -1]]

   # Sweep dxs
   for ii in range(len(xrang)):
       nx = xrang[ii]
       x = np.linspace(0, XF, nx) ; dx = x[1] - x[0]
       dxs.append(dx)

       # Run solver, hold dt fixed
       u = geturec(nu=nu, x=x, diffstrategy=diffstrategy, convstrategy=convstrategy, evolution_time=.02, n_save_t=20, ubl=0, ubr=0, dt=truedt)
       #if ii == 0: plt.plot(u) ; plt.show()
       us.append(u[:, -1])

       # record |L-1(dx) - L-1(truth)|
       print (ii, u[:, -1].size, trueu[:, -1].size, trueu[1 :: 2 ** (ii+1), -1].size)
       print(u[:, -1])
       print('----')
       print(u[:, -1] - trueu[0 :: 2 * 2 ** (ii), -1])
       print('====')
       qoi = np.sqrt(np.sum((u[:, -1] - trueu[0 :: 2 * 2 ** (ii), -1]) ** 2) / nx)
       #qoi = ul1(u[:, -1], dx)
       ypoints.append(qoi)
       #ypoints.append(np.abs(trueqoi - qoi))

   return dxs, ypoints, us

# Plot results. The fourth order method should have a slope of 4 on the log-log plot.
from scipy.optimize import minimize as mini
strategy = 4
if strategy == 2:
    diffstrat = '3c'
    convstrat = '2c'
elif strategy == 4:
    diffstrat = '5c'
    convstrat = '4c'
else: raise(Exception("Error"))
dxs, ypoints, us = errf(diffstrat, convstrat)
def fit2(a): return 1000 * np.sum((a * np.array(dxs) ** 2 - ypoints) ** 2)
def fit3(a): return 1000 * np.sum((np.exp(a) * np.array(dxs) ** 3 - ypoints) ** 2)
def fit4(a): return 100000000 * np.sum((np.exp(a) * np.array(dxs[0]) ** 4 - ypoints[0]) ** 2)
amin = 99
for ii in [1e-4, 1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
   a = mini(fit2, ii)
   if a.fun < amin:
      amin = a.fun
      if a.x > 0: ax = a.x
bmin = 99
for ii in [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2 ,3 ,4, 6, 8, 9, 11, 13]:
   b = mini(fit3, ii)
   if b.fun < bmin:
      bmin = b.fun
      bx = b.x
cmin = 99
for ii in [-4, -3, -2, -1, 0, 1, 2, 3 ,4, 6, 8, 9, 11, 13]:
   c = mini(fit4, ii)
   if c.fun < cmin:
      cmin = c.fun
      cx = c.x
plt.plot(dxs,  ax * np.array(dxs)**2, c='k', label=r"$\Delta x^2$", ls='--')
plt.plot(dxs,  np.exp(bx) * np.array(dxs)**3, c='k', label=r"$\Delta x^3$", ls='-.')
plt.plot(dxs, np.exp(cx) * np.array(dxs)**4, c='k', label=r"$\Delta x^4$")
plt.plot(dxs, ypoints, label=r"Convergence", marker='x')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$\Delta X$")
plt.ylabel("$|L-L_{true}|$")
plt.title(r"$\nu=%f, strategy=%s$"%(nu, strategy))
plt.legend()
plt.savefig('%s_diditwork.pdf'%strategy, bbox_inches='tight')
