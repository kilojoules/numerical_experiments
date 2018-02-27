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

@nb.jit
def geturec(nu=.05, x=x, evolution_time=evolution_time, u0=0, n_save_t=500, ub=1):

    # Prescribde c=.1 and cfts=.3
    dx = x[1] - x[0]
    dt = min(.1 * dx / 20., .3 / nu * dx ** 2)
    nt = int(evolution_time / dt)
    divider = nt / n_save_t
    if divider ==0: raise(IOError("not enough time steps to save %i times"%n_save_t))

    # initially purturb u
    u_initial = ub + .8 * np.sin(x)
    if u0: u_initial = u0
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

        # Dirchlet BCs
        u[1:-1] = un[1:-1] + dt *(-1 * un[1:-1] * (un[1:-1] - un[:-2]) / dx + nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2 )

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


