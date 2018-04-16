import numpy as np
import matplotlib.pyplot as plt
import numba as nb
XF = 1.

# Convective Differentiation function, approximates du/dx
#@nb.jit
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
        # I made this negative negative \|/
        duconv[-2] = un[-2] * ( - 25./12. * un[-2] + 4 * un[-3] - 3 * un[-4] + 4./3. * un[-5] - un[-6]/4.) / dx
        duconv[2:-2] = -1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx)
    return duconv

# Diffustive Differentiation function, approximates nu d^2 u /dx^2
#@nb.jit
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
#@nb.jit
def geturec(x, nu=.05, evolution_time=1, u0=None, n_save_t=50, ubl=0., ubr=0., diffstrategy='5c', convstrategy='4c', dt=None, returndt=False):

    dx = x[1] - x[0]

    # Prescribde cfl=0.05 and ftcs=0.02
    if dt is not None: pass
    else: dt = min(.2 * dx / 1., .2 / nu * dx ** 2)
    if returndt: return dt

    # Determine the interval, "divider", to record time with
    nt = int(evolution_time / dt)
    divider = int(nt / float(n_save_t))
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # The default initial condition is a half sine wave.
    u_initial = ubl + np.sin(x)
    #u_initial = ubl + np.sin(x * np.pi)
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
        #dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy) 
        dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy) + convective_dudt(un, dx, strategy=convstrategy)

        # forward euler time step
        u = un + dt * dudt

        # RK 4 time step
        #uhalfn = un + dt * dudt / 2.
        #duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy)
        ##uhalfk2 = un + duhalfn_dt1 * dt / 2
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

if __name__=="__main__":
     record = {}
     nus = [1, .5, .1, .05, .01, .005, .001, .0005, .0001]
     for ET in [.1, .5, 1.]:
     #for ET in [.05, .1, .5, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4]:
        record[ET] = []
        for nu in nus:
           nx = 25
           oldu = None
           while True:
              x = np.linspace(0, np.pi, nx + 1)
              u = geturec(x, nu=nu, evolution_time=ET, n_save_t=1)[:, -1]
              if oldu is not None:
                 #qoi = np.max((u[0::2] - oldu)**2) / (nx/2)
                 #qoi = np.sqrt(np.sum((u[0::2] - oldu)**2 / (nx/2) * u[0::2] ** 2))
                 qoi = np.sqrt(np.sum((u[0::2] - oldu)**2) / (nx/2))
                 #qoi = np.sqrt(np.sum((u[0::2] - oldu)**2) / (nx/2) * np.mean(u))
                 print(ET, nu, nx, qoi)
                 if qoi < 1e-4:
                    record[ET].append(nx)
                    break
              nx *= 2
              oldu = u
     for ET in record.keys(): plt.plot(nus, record[ET], label=ET, marker='x')
     leg = plt.legend()
     leg.set_title('Evolution Time')
     plt.ylabel('# x points')
     plt.xlabel(r'$\nu$')
     plt.xscale('log')
     plt.yscale('log')
     plt.savefig('ETs.pdf')
        #fl = open('mine.dat', 'w')
    ##l.write(' '.join([str(s) for s in u[:, -1]]))
    #fl.close()
