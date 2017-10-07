import numpy as np
import numba as nb
from matplotlib import pyplot as plt

nx = 41
ny = 41
nit = 500 # Number of iterations for pressure eqn
nt = 4000 # Number of outer iterations
c = 1
dx = 2 / (nx - 1.)
dy = 2 / (ny - 1.)
x = np.linspace(0., 2, nx)
y = np.linspace(0., 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

Pr = 1e-4
Ra = 1e-4

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
phi = np.zeros((ny, nx)) 
b = np.zeros((ny, nx))

# helper function. These terms are part of the posson equation.
@nb.jit(parallel=True)
def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

@nb.jit(parallel=True)
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] ##dp/dy = 0 at x = 2
        p[0, :] = p[1, :]  ##dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    ##dp/dx = 0 at x = 0
        p[-1, :] = 0        ##p = 0 at y = 2
        
    return p

@nb.jit(parallel=True)
def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, phi):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        phin = phi.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        #p = rho * un[:, :] ** 2 + vn[:, :] ** 2

        #phi[1:-1, 1:-1] = phin[1:-1, 1:-1] + dt * (1/np.sqrt(Ra) * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]) / dx / dx + (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) / dy / dy  - (u[1:-1, 1:-1] * phin[1:-1, 1:-1] - u[0:-2, 1:-1] * phin[0:-2, 1:-1]) / dx - (v[1:-1, 1:-1] * phin[1:-1, 1:-1] - v[1:-1, 0:-2] * phin[1:-1, 0:-2]) / dy)
        
        #u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * ( -1 * (p[1:-1, 1:-1] - p[0:-2, 1:-1])/dx + Pr / np.sqrt(Ra) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx / dx - un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dx - vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dy )

        #v[1:-1,1:-1] = vn[1:-1, 1:-1] + dt * ( -1 * (p[1:-1, 1:-1] - p[1:-1, 0:-2])/dy + Pr/np.sqrt(Ra) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dy / dy - un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dx - vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dy ) + phi[1:-1,1:-1] * Pr

        phi[1:-1, 1:-1] = (phin[1:-1, 1:-1] + 
                          dt * ( 
                            .001 * ((phin[2:, 1:-1] - 2 * phin[1:-1, 1:-1] + phin[0:-2, 1:-1]) / dy / dy + 
                            (phin[1:-1, 2:] - 2 * phin[1:-1, 1:-1] + phin[1:-1, 0:-2]) / dx / dx ) -
                            v[1:-1, 1:-1] *(phin[1:-1, 1:-1] - phin[0:-2, 1:-1]) / dy - 
                            u[1:-1, 1:-1] *(phin[1:-1, 1:-1] - phin[1:-1, 0:-2]) / dx))
        phi[0:,0:2] = 1.1
        #phi[0, :] = 1
        #phi[0:,0:10] = 1.1        

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))) #+ Pr * (phin[1:-1, 1:-1] - .2)


        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    #set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :]=0
        v[:, 0] = 0
        v[:, -1] = 0
        
        
    return u, v, p, phi


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
phi = np.ones((ny, nx)) #+ np.random.normal(5, 1, (ny,nx))
phi[0:,0:2] = 1.1
u, v, p, phi = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, phi)


fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
c = plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.viridis)  
cb = fig.colorbar(c)
cb.set_label('Pressure')
# plotting the temperature field outlines
c = plt.contour(X, Y, phi, cmap=plt.cm.coolwarm)  
cb = fig.colorbar(c)
cb.set_label('Temperature')
# plotting velocity field
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
