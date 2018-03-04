# Load modules
import numpy as np
import numba as nb

# Let's evolve burgers' equation in time

# x domain
nx = 1000
xf = np.pi
x = np.linspace(0., xf, nx)
dx = x[1] - x[0]

# when should I evolve it to (default)
evolution_time = 1

#@nb.jit
def geturec(nu=.05, x=x, evolution_time=evolution_time, u0=None, n_save_t=500, ub=1, strategy='4c'):

    # Prescribde c=.1 and cfts=.3
    dx = x[1] - x[0]
    dt = min(.1 * dx / 20., .3 / nu * dx ** 2)
    nt = int(evolution_time / dt)
    divider = nt / n_save_t
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # initially purturb u
    u_initial = ub + .8 * np.sin(x)
    if u0 is not None: u_initial = u0
    #u_initial = 1 + .2 * np.sin(x) #+ 0.3 * np.sin(x * 3)
    u = u_initial.copy()
    u[[0, -1]] = ub

    u0 = u.copy()


    # u_record holds all the snapshots. 
    #if divider == 0: u_record = np.zeros((x.size, 1))
    #else: u_record = np.zeros((x.size, nt / divider + 2))
    u_record = np.zeros((x.size, nt / divider + 2))

    # Evolve through time
    ii = 1
    u_record[:, 0] = u.copy()
    for _ in range(nt):
    #for _ in nb.prange(nt): - BEWARE
        un = u.copy()

        # Dirchlet BCs, 2-term-upwind convective, 3-term central viscous
        if strategy == '2u':
            u[1:-1] = un[1:-1] + dt *(-1 * un[1:-1] * (un[1:-1] - un[:-2]) / dx + nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )

        # Dirchlet BCs, 2-term central convective, 3-term central viscous
        elif strategy == '2c':
            u[1:-1] = un[1:-1] + dt *(-1 * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx) + nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )

        # Dirchlet BCs, 3-term-upwind convective, 3-term central viscous
        elif strategy == '3u':
            u[1] = un[1] + dt *(-1 * un[1] * (un[1] - un[0]) / dx + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            u[-2] = un[-2] + dt *(-1 * un[-2] * (un[-1] - un[-2]) / dx + nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2 )
            u[2:-2] = un[2:-2] + dt *(-1 * un[2:-2] * (3 * un[2:-2] - 4 *  un[1:-3] + un[:-4]) / (2 * dx) + nu * (un[3:-1] - 2 * un[2:-2] + un[1:-3]) / dx**2 )

        # Dirchlet BCs, 4-term-central convective, 3-term central viscous
        elif strategy == '4c':
            u[1] = un[1] + dt *(-1 * un[1] * (un[2] - un[0]) / (2 * dx) + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            u[-2] = un[-2] + dt *(-1 * un[-2] * (un[-1] - un[-3]) / (2 * dx) + nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2 )
            u[2:-2] = un[2:-2] + dt *(-1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx ) + nu * (un[3:-1] - 2 * un[2:-2] + un[1:-3]) / dx**2 )

        # Dirchlet BCs, Lax-Wendroff Method
        elif strategy == 'lw':
            ustarm2 = un[1:-5] - dt/dx * ((nu * (un[3:-3] - un[1:-5]) / (2 * dx) - 0.5 * un[2:-4] ** 2) - (nu * (un[2:-4] - un[:-6]) / (2 * dx) - 0.5 * un[1:-5] ** 2))
            ustarm1 = un[2:-4] - dt/dx * ((nu * (un[4:-2] - un[2:-4]) / (2 * dx) - 0.5 * un[3:-3] ** 2) - (nu * (un[3:-3] - un[1:-5]) / (2 * dx) - 0.5 * un[2:-4] ** 2))
            ustar0  = un[3:-3] - dt/dx * ((nu * (un[5:-1] - un[3:-3]) / (2 * dx) - 0.5 * un[4:-2] ** 2) - (nu * (un[4:-2] - un[2:-4]) / (2 * dx) - 0.5 * un[3:-3] ** 2))
            ustar1  = un[4:-2] - dt/dx * ((nu * (un[6:] - un[4:-2]) / (2 * dx) - 0.5 * un[5:-1] ** 2) - (nu * (un[5:-1] - un[3:-3]) / (2 * dx) - 0.5 * un[4:-2] ** 2))
            #u[1] =  un[1]  + dt *(-1 * un[1] *  (un[2] - un[0]) / (2 * dx) + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            #u[2] =  un[2]  + dt *(-1 * un[2] *  (un[3] - un[1]) / (2 * dx) + nu * (un[3] - 2 * un[2] + un[1]) / dx**2 )
            #u[-2] = un[-2] + dt *(-1 * un[-2] * (un[-1] - un[-3]) / (2 * dx) + nu * (un[-3] - 2 * un[-2] + un[-1]) / dx**2 )
            #u[-3] = un[-3] + dt *(-1 * un[-3] * (un[-2] - un[-4]) / (2 * dx) + nu * (un[-4] - 2 * un[-3] + un[-2]) / dx**2 )
            u[1] = un[1] + dt *(-1 * un[1] * (un[1] - un[0]) / dx + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            u[2] = un[2] + dt *(-1 * un[2] * (un[2] - un[1]) / dx + nu * (un[3] - 2 * un[2] + un[1]) / dx**2 )
            u[-3] = un[-3] + dt *(-1 * un[-3] * (un[-2] - un[-3]) / dx + nu * (un[-2] - 2 * un[-3] + un[-4]) / dx**2 )
            u[-2] = un[-2] + dt *(-1 * un[-2] * (un[-1] - un[-2]) / dx + nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2 )
            u[3:-3] = 0.5 * (un[3:-3] + ustar0) - dt / 2 / dx * (( nu * (ustar1 - ustarm1) / (2 * dx) - .5 * ustar0 ** 2 ) - (nu * (ustar0 - ustarm2) / (2 * dx) - 0.5 * ustarm1 ** 2))

        elif strategy == 'rlw':
            um1 = 0.5 * (un[3:-3] + un[4:-2]) - dt / dx / 2 * (((un[5:-1] - un[3:-3])/(2 * dx) - 0.5 * un[4:-2] ** 2) - ((un[4:-2] - un[2:-4]) / (2 * dx) - un[3:-3] ** 2))
            um2 = 0.5 * (un[2:-4] + un[3:-3]) - dt / dx / 2 * (((un[4:-2] - un[2:-4])/(2 * dx) - 0.5 * un[3:-3] ** 2) - ((un[3:-3] - un[1:-5]) / (2 * dx) - un[2:-4] ** 2))
            um3 = 0.5 * (un[1:-5] + un[2:-4]) - dt / dx / 2 * (((un[3:-3] - un[1:-5])/(2 * dx) - 0.5 * un[2:-4] ** 2) - ((un[2:-4] - un[:-6]) / (2 * dx) - un[1:-5] ** 2))
            u[3:-3] = un[3:-3] - dt / dx * ((nu * (um1 - um2) / 2 / dx - um1 ** 2 ) - (nu * (um2 - um3) / 2 / dx - um2 ** 2))
            
        elif strategy == 'rk':
            u[1] = un[1] + dt *(-1 * un[1] * (un[2] - un[0]) / (2 * dx) + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            u[-2] = un[-2] + dt *(-1 * un[-2] * (un[-1] - un[-3]) / (2 * dx) + nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2 )
            uh = un.copy()
            uh[1] = un[1] + dt/2 *(-1 * un[1] * (un[2] - un[0]) / (2 * dx) + nu * (un[2] - 2 * un[1] + un[0]) / dx**2 )
            uh[-2] = un[-2] + dt/2 *(-1 * un[-2] * (un[-1] - un[-3]) / (2 * dx) + nu * (un[-1] - 2 * un[-2] + un[-3]) / dx**2 )
            uh[2:-2] = un[2:-2] + dt / 2 *(-1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx ) + nu * (un[3:-1] - 2 * un[2:-2] + un[1:-3]) / dx**2 )
            u[2:-2] = un[2:-2] + .5 * dt *(-1 * un[2:-2] * (-1 * un[4:] + 8 * un[3:-1]  - 8 * un[1:-3] + un[:-4]) / ( 12 * dx ) + nu * (un[3:-1] - 2 * un[2:-2] + un[1:-3]) / dx**2 -1 * uh[2:-2] * (-1 * uh[4:] + 8 * uh[3:-1]  - 8 * uh[1:-3] + uh[:-4]) / ( 12 * dx ) + nu * (uh[3:-1] - 2 * uh[2:-2] + uh[1:-3]) / dx**2 )
           
        else: raise(IOError("Bad Strategy"))


        # Periodic BCs
        #u[1:-1] = un[1:-1] + dt *( -1 * un[1:-1] * (un[1:-1] - un[:-2]) / dx + nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )
        #u[-1] = un[-1] +  dt * ( -1 * un[-1] * (un[-1] - un[-2]) / dx + nu * (un[0] - 2 * un[-1] + un[-2]) / dx ** 2)
        #u[0] = un[0] +  dt * ( -1 * un[0] * (un[0] - un[-1]) / dx + nu * (un[1] - 2 * un[0] + un[-1]) / dx ** 2)
   
        # Save every xth time step
        if _ % (divider) == 0:
            #print "C # ---> ", u * dt / dx
            u_record[:, ii] = u.copy()
            ii += 1
    u_record[:, -1] = u
    return u_record

if __name__=='__main__': u_record = geturec()


