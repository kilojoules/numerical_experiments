import numpy as np
import numba as nb
XF = 1.

# Convective Differentiation function, approximates du/dx
#@nb.jit
def convective_dudt(un, dx, strategy='4c'):
    duconv = np.zeros(un.size)
    #duconv = np.zeros(un.size, dtype=np.float128)
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
        duconv[2:-2] = -1 * un[2:-2] * (-1./12. * un[4:] + 8./12. * un[3:-1]  - 8/12. * un[1:-3] + 1./12. * un[:-4]) / (dx)
        #duconv[2:-2] = -1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx)
    return duconv

# Diffustive Differentiation function, approximates nu d^2 u /dx^2
#@nb.jit
def diffusive_dudt(un, nu, dx, strategy='5c'):
    dundiff = np.zeros(un.size)
    #dundiff = np.zeros(un.size, dtype=np.float128)

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
def geturec(x, nu=.05, evolution_time=1, u0=None, n_save_t=50, ubl=0., ubr=0., diffstrategy='5c', convstrategy='4c', timestrategy='fe', dt=None, returndt=False):

    dx = x[1] - x[0]

    # Prescribde cfl=0.05 and ftcs=0.02
    if dt is not None: pass
    else: dt = min(.02 * dx / 1., .02 / nu * dx ** 2)
    if returndt: return dt

    # Determine the interval, "divider", to record time with
    nt = int(evolution_time / dt)
    dt = evolution_time / nt
    print('t is ', nt * dt)
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
    #u_record = np.zeros((x.size, int(nt / divider + 2)),  dtype=np.float128)

    # Evolve through time
    ii = 1
    u_record[:, 0] = u
    for _ in range(nt):
        un = u.copy()
        #dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy) 
        dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy) + convective_dudt(un, dx, strategy=convstrategy)

        # forward euler time step
        if timestrategy == 'fe':
           u = un + dt * dudt
        elif timestrategy == 'rk2':
            uhalfn = un + dt * dudt / 2.
            duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy) + convective_dudt(uhalfn, dx, strategy=convstrategy)
            u = un + dt * duhalfn_dt1
            #u = 0.5 * (un + dt * dudt + uhalfn + duhalfn_dt1 * dt)
            if _ == 0: print('hey!')

        # RK 4 time step
        elif timestrategy == 'rk4':
            uhalfn = un + dt * dudt / 2.
            duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy) + convective_dudt(uhalfn, dx, strategy=convstrategy)
            uhalfk2 = un + duhalfn_dt1 * dt / 2
            duhalfk2_dt = diffusive_dudt(uhalfk2, nu, dx, strategy=diffstrategy) + convective_dudt(uhalfk2, dx, strategy=convstrategy)
            ufull = un + duhalfk2_dt * dt
            dufull_dt = diffusive_dudt(ufull, nu, dx, strategy=diffstrategy)+ convective_dudt(ufull, dx, strategy=convstrategy)
            u = un + (dt / 6.) * (dudt + 2 * duhalfn_dt1 + 2 * duhalfk2_dt + dufull_dt)
        else: raise(Exception("Error"))

        # Save every mth time step
    #return u
        if _ % divider == 0:
            u_record[:, ii] = u.copy()
            ii += 1
    u_record[:, -1] = u
    return u_record
    #return u_record[1:-1, :]

if __name__=="__main__":
    x = np.linspace(0, np.pi, 801)
    u = geturec(x, nu=0.1, dt=5e-9, evolution_time=0.00002, n_save_t=1)[:, -1]
    fl = open('mine.dat', 'w')
    fl.write(' '.join([str(s) for s in u]))
    #fl.write(' '.join([str(s) for s in u[:, -1]]))
    fl.close()
