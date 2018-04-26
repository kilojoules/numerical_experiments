import numpy as np
import numba as nb
XF = 1.

# Convective Differentiation function, approximates du/dx
#@nb.jit
def convective_dudt(un, dx, strategy='4c', BCs='dirchlet'):
    duconv = np.zeros(un.size)
    if BCs == 'dirchlet':
        if strategy=='2c':
            duconv[1:-1] = -1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx)
        elif strategy == '4c':
            #duconv[2:-2] = -1 * un[2:-2] * (-1./12. * un[4:] + 8./12. * un[3:-1]  - 8/12. * un[1:-3] + 1./12. * un[:-4]) / (dx)
            #duconv[1] = -1 * un[1] * ( - 25./12. * un[1] + 4 * un[2] - 3 * un[3] + 4./3. * un[4] - un[5]/4.) / dx
            #duconv[-2] = un[-2] * ( - 25./12. * un[-2] + 4 * un[-3] - 3 * un[-4] + 4./3. * un[-5] - un[-6]/4.) / dx
            duconv[1] = -1 * un[1] * ( - 25./12. * un[1] + 4 * un[2] - 3 * un[3] + 4./3. * un[4] - un[5]/4.) / dx
            duconv[-2] = un[-2] * ( - 25./12. * un[-2] + 4 * un[-3] - 3 * un[-4] + 4./3. * un[-5] - un[-6]/4.) / dx
            duconv[2:-2] = -1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx)


            #duconv[1] = -1 * un[1] * ( -12./60. * un[0] - 65./60. * un[1] + 120./60. * un[2] - 60./60. * un[3] + 20./60. * un[4] - 3./60. * un[5]) / dx
            #duconv[-2] = un[-2] * ( -12./60. * un[-1] - 65./60. * un[-2] + 120./60. * un[-3] - 60./60. * un[-4] + 20./60. * un[-5] - 3./60. * un[-6]) / dx
        else: raise(IOError("Invalid convective strategy")) ; quit()
    elif BCs == 'periodic':
        if strategy=='2c':
            duconv[1:-1] = -1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx)
            duconv[0] = -1 * un[0] * (un[1] - un[-1]) / (2 * dx)
            duconv[-1] = -1 * un[-1] * (un[0] - un[-2]) / (2 * dx)
        elif strategy == '4c':
            duconv[2:-2] = -1 * un[2:-2] * (-1./12. * un[4:] + 8./12. * un[3:-1]  - 8/12. * un[1:-3] + 1./12. * un[:-4]) / (dx)
            duconv[1] = -1 * un[1] * ( -1./12. * un[3] + 8./12. * un[2] - 8./12. * un[0] + 1./12. * un[-1]) / dx
            duconv[0] = -1 * un[0] * ( -1./12. * un[2] + 8./12. * un[1] - 8./12. * un[-1] + 1./12. * un[-2]) / dx
            duconv[-1] = -1 * un[-1] * ( -1./12. * un[1] + 8./12. * un[0] - 8./12. * un[-2] + 1./12. * un[-3]) / dx
            duconv[-2] = -1 * un[-2] * ( -1./12. * un[0] + 8./12. * un[-1] - 8./12. * un[-3] + 1./12. * un[-4]) / dx
        else: raise(IOError("Invalid convective strategy")) ; quit()
    else: raise(IOError("Invalid convective BC")) ; quit()
    return duconv

# Diffustive Differentiation function, approximates nu d^2 u /dx^2
#@nb.jit
def diffusive_dudt(un, nu, dx, strategy='5c', BCs='dirchlet'):
    dundiff = np.zeros(un.size)

    if BCs == 'dirchlet':
        if strategy == '3c':
            dundiff[1:-1] = nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2
        elif strategy == '5c':
            dundiff[1] = nu * (15./4. * un[1] - 77./6. * un[2] + 107./6. * un[3] - 13 * un[4] + (61./12.) * un[5] - 5./6. * un[6]) / dx ** 2
            dundiff[-2] = nu * (15./4. * un[-2] - 77./6. * un[-3] + 107./6. * un[-4] - 13 * un[-5] + (61./12.) * un[-6] - 5./6. * un[-7]) / dx ** 2
            dundiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )
            #dundiff[1] = nu * (15./4. * un[1] - 77./6. * un[2] + 107./6. * un[3] - 13 * un[4] + (61./12.) * un[5] - 5./6. * un[6]) / dx ** 2
            #dundiff[-2] = nu * (15./4. * un[-2] - 77./6. * un[-3] + 107./6. * un[-4] - 13 * un[-5] + (61./12.) * un[-6] - 5./6. * un[-7]) / dx ** 2
            #dundiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )

            #dundiff[1] = nu * (137./180. * un[0] - 147./180. * un[1] - 255./180. * un[2] + 470./180. * un[3] - 285./180. * un[4] + 93./180. * un[5] - 13./180. * un[6]) / dx ** 2
            #dundiff[-2] = nu * (137./180. * un[-1] - 147./180. * un[-2] - 255./180. * un[-3] + 470./180. * un[-4] - 285./180. * un[-5] + 93./180. * un[-6] - 13./180. * un[-7]) / dx ** 2
        else: raise(IOError("Invalid diffusive strategy")) ; quit()
    elif BCs == 'periodic':
        if strategy == '3c':
            dundiff[1:-1] = nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2
            dundiff[-1] = nu * (un[0] - 2 * un[-1] + un[-2]) / dx**2
            dundiff[0] = nu * (un[1] - 2 * un[0] + un[-1]) / dx**2
        elif strategy == '5c':
            dundiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )
            dundiff[1] = nu * (-1 * un[3] + 16 * un[2] - 30 * un[1] + 16 * un[0] - un[-1]) / (12 * dx ** 2)
            dundiff[0] = nu * (-1 * un[2] + 16 * un[1] - 30 * un[0] + 16 * un[-1] - un[-2]) / (12 * dx ** 2)
            dundiff[-1] = nu * (-1 * un[1] + 16 * un[0] - 30 * un[-1] + 16 * un[-2] - un[-3]) / (12 * dx ** 2)
            dundiff[-2] = nu * (-1 * un[0] + 16 * un[-1] - 30 * un[-2] + 16 * un[-3] - un[-4]) / (12 * dx ** 2)
        else: raise(IOError("Invalid diffusive strategy")) ; quit()
    else: raise(IOError("Invalid diffusive BC")) ; quit()
    return dundiff

# Velocity Evolution Function. Accepts initial and boundary conditions, returns time evolution history.
#@nb.jit
def geturec(x, nu=.05, evolution_time=1, u0=None, n_save_t=50, ubl=0., ubr=0., diffstrategy='5c', convstrategy='4c', timestrategy='fe', dt=None, returndt=False, BCs='periodic'):

    dx = x[1] - x[0]

    # Prescribde cfl=0.05 and ftcs=0.02
    if dt is not None: pass
    else: dt = min(.02 * dx / 1., .02 / nu * dx ** 2)
    if returndt: return dt

    # Determine the interval, "divider", to record time with
    nt = int(evolution_time / dt)
    dt = evolution_time / nt
    divider = int(nt / float(n_save_t))
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # The default initial condition is a half sine wave.
    u_initial = ubl + np.sin(x) ** 2 
    #u_initial = ubl + np.sin(x * np.pi)
    if u0 is not None: u_initial = u0
    u = u_initial
    if BCs == 'dirchlet':
        u[0] = ubl
        u[-1] = ubr

    # insert ghost cells; extra cells on the left and right
    # for the edge cases of the finite difference scheme
    #if BCs == 'periodic':
    #   x = np.insert(x, 0, x[0]-dx)
    #   x = np.insert(x, -1, x[-1]+dx)
    #   u = np.insert(u, 0, ubl)
    #   u = np.insert(u, -1, ubr)

    # u_record holds all the snapshots. They are evenly spaced in time,
    # except the final and initial states
    u_record = np.zeros((x.size, int(nt / divider + 2)))
    #u_record = np.zeros((x.size, int(nt / divider + 2)),  dtype=np.float128)

    # Evolve through time
    ii = 1
    u_record[:, 0] = u
    for _ in range(nt):
        un = u.copy()
        #dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy, BCs=BCs) 
        #dudt = convective_dudt(un, dx, strategy=convstrategy, BCs=BCs)
        #print(diffusive_dudt(un, nu, dx, strategy=diffstrategy, BCs=BCs))
        dudt = diffusive_dudt(un, nu, dx, strategy=diffstrategy, BCs=BCs) + convective_dudt(un, dx, strategy=convstrategy, BCs=BCs)

        # forward euler time step
        if timestrategy == 'fe':
           u = un + dt * dudt
        elif timestrategy == 'rk2':
            uhalfn = un + dt * dudt / 2.
            duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy, BCs=BCs) + convective_dudt(uhalfn, dx, strategy=convstrategy, BCs=BCs)
            u = un + dt * duhalfn_dt1
            #u = 0.5 * (un + dt * dudt + uhalfn + duhalfn_dt1 * dt)

        # RK 4 time step
        elif timestrategy == 'rk4':
            uhalfn = un + dt * dudt / 2.
            duhalfn_dt1 = diffusive_dudt(uhalfn, nu, dx, strategy=diffstrategy, BCs=BCs) + convective_dudt(uhalfn, dx, strategy=convstrategy, BCs=BCs)
            uhalfk2 = un + duhalfn_dt1 * dt / 2
            duhalfk2_dt = diffusive_dudt(uhalfk2, nu, dx, strategy=diffstrategy, BCs=BCs) + convective_dudt(uhalfk2, dx, strategy=convstrategy, BCs=BCs)
            ufull = un + duhalfk2_dt * dt
            dufull_dt = diffusive_dudt(ufull, nu, dx, strategy=diffstrategy, BCs=BCs) + convective_dudt(ufull, dx, strategy=convstrategy, BCs=BCs)
            u = un + (dt / 6.) * (dudt + 2 * duhalfn_dt1 + 2 * duhalfk2_dt + dufull_dt)
        else: raise(Exception("Time Error"))

        # Save every mth time step
    #return u
        if _ % divider == 0:
            u_record[:, ii] = u.copy()
            ii += 1
    u_record[:, -1] = u
    return u_record

if __name__=="__main__":
    x = np.linspace(0, np.pi, 801)
    u = geturec(x, nu=0.1, dt=5e-9, evolution_time=0.00002, n_save_t=1)[:, -1]
    fl = open('mine.dat', 'w')
    fl.write(' '.join([str(s) for s in u]))
    #fl.write(' '.join([str(s) for s in u[:, -1]]))
    fl.close()
