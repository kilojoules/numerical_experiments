# Load modules
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# Let's evolve burgers' equation in time

# x domain
nx = 1000
xf = np.pi
x = np.linspace(0., xf, nx)
dx = x[1] - x[0]

# when should I evolve it to (default)
evolution_time = 0.5

@nb.jit
def convective_dudt(un, dx, strategy='4c'):
     ###############
    # convective dudt
     ###############
    duconv = np.zeros(un.size, dtype=np.float128)
    # Dirchlet BCs, 2-term-upwind convective
    if strategy == '2u':
        duconv[1:-1] = -1 * un[1:-1] * (un[1:-1] - un[:-2]) / dx

    # Dirchlet BCs, 2-term central convective
    elif strategy == '2c':
        duconv[1:-1] = -1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx)

    # Dirchlet BCs, 3-term-upwind convective
    elif strategy == '3u':
        duconv[1] = -1 * un[1] * (un[1] - un[0]) / dx 
        duconv[-2] = -1 * un[-2] * (un[-1] - un[-2]) / dx 
        duconv[2:-2] = -1 * un[2:-2] * (3 * un[2:-2] - 4 *  un[1:-3] + un[:-4]) / (2 * dx)

    # Dirchlet BCs, 4-term-central convective
    elif strategy == '4c':
        duconv[1] = -1 * un[1] * (un[2] - un[0]) / (2 * dx)
        duconv[-2] = -1 * un[-2] * (un[-1] - un[-3]) / (2 * dx)
        duconv[2:-2] = -1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx)
    else: raise(IOError("Invalid convective strategy")) ; quit()
    return duconv


@nb.jit
def diffusive_dudt(un, nu, dx, strategy='5c'):
     ###############
    # diffusive dudt
     ###############
    dundiff = np.zeros(un.size, dtype=np.float128)

    # Dirchlet BCs, 5-term central viscous
    if strategy == '3c':
        dundiff[1:-1] = nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2
        #print("!!!!!")
        #print((un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )
        #print("!!!!!")

    elif strategy == '5c':
        dundiff[1] = nu * (un[2] - 2 * un[1] + un[0]) / dx**2
        dundiff[-2] = nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2
        dundiff[2:-2] = nu * (-1 * un[4:] + 16 * un[3:-1] - 30 * un[2:-2] + 16 * un[1:-3] - un[:-4]) / (12 * dx**2 )
    else: raise(IOError("Invalid diffusive strategy")) ; quit()
    return dundiff

@nb.jit
def geturec(nu=.05, x=x, evolution_time=evolution_time, u0=None, n_save_t=500, ubl=1., ubr=1., convstrategy="4c", diffstrategy='5c', timestrategy='rk'):

    # Prescribde cfl and cfts=.05
    dx = x[1] - x[0]
    #dt = min(.3 * dx / 3., .3 / nu * dx ** 2)
    dt = min(.1 * dx / 20., .3 / nu * dx ** 2)

    #dt = min(.05 * dx / 20., .05 / nu * dx ** 2)
    #print(dt, dx, nu) ; hey
    nt = int(evolution_time / dt)
    divider = int(nt / float(n_save_t))
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # initially purturb u
    u_initial = ubl + .8 * np.sin(x)
    if u0 is not None: u_initial = u0
    #u_initial = 1 + .2 * np.sin(x) #+ 0.3 * np.sin(x * 3)
    u = u_initial
    u[0] = ubl
    u[-1] = ubr

    u0 = u

    # u_record holds all the snapshots. 
    #if divider == 0: u_record = np.zeros((x.size, 1))
    #else: u_record = np.zeros((x.size, nt / divider + 2))
    u_record = np.zeros((x.size, int(nt / divider + 2)))

    # Evolve through time
    ii = 1
    u_record[:, 0] = u
    for _ in range(nt):
    #for _ in nb.prange(nt): - BEWARE
        un = u.copy()
        #plt.plot(u) ; plt.show()
        duconv = convective_dudt(un, dx, strategy=convstrategy)
        dudiff = diffusive_dudt(un, nu, dx, strategy=diffstrategy)
        
        #print('====')
        #print((un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )
        #print(dudiff[1:-1] )
      
        #print('-----')
        #print(nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )
        #print(duconv * dt)
        #if _ == 1: hey

        # scheme is to compute du/dt in two parts: convective and diffusive.

        # Combine convective and diffusive dudt terms to get full derivative
        dudt = duconv + dudiff
        #print('*********')
        #print(dudt)
        #print('====')

        # time strategies

        #u[1:-1] = un[1:-1] + dt *(-1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx) + nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )
        #if True: pass
        if timestrategy == 'vanilla':
            u = un + dt * dudt
        elif timestrategy == 'rk':
            uhalfn = un + dt * dudt / 2.
            duhalfn_dt = convective_dudt(uhalfn, strategy=convstrategy) + diffusive_dudt(uhalfn, nu, strategy=diffstrategy)
            u = un + 0.5 * dt * (dudt + duhalfn_dt)
        else: raise(IOError("Bad Strategy"))

        # Save every xth time step
        if _ % divider == 0:
            #print "C # ---> ", u * dt / dx
            u_record[:, ii] = u.copy()
            ii += 1
    u_record[:, -1] = u
    return u_record

if __name__=='__main__': u_record = geturec()


