### Test File

import backward_euler as bc
import crank_nicolson as cn
import numpy as np
import pylab as pl
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import pi

"""

###------------------------------------------------------###


# Problem 1:  (Homogeneous Dirichelt BCs)

#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0


# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5        # total time to solve for

def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# set numerical parameters
mx = 100    # number of gridpoints in space
mt = 1000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]

#u_j = bc.backward_euler_solver_dirichlet(u_I,params)
u_j = cn.crank_nicolson_solver_homogeneous_BC(u_I,params)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()



### --------------------------------------------------------------###


# Problem 2:    (non-homogeneous Dirichelt BCs)

#   u_t = kappa u_xx  0<x<L, 0<t<T
# with temperature boundary conditions
#   u(0,t) = constant_1, u(L,t) = constant_2
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0


# set problem parameters/functions
kappa = 9.0   # diffusion constant
L=2.0         # length of spatial domain
T=0.5        # total time to solve for

left_BC_fun = lambda t:  0*t+0
right_BC_fun = lambda t: 0*t+8

def u_I(x):
    # initial temperature distribution
    y = 2*x**2
    return y

def u_exact(x,t):

    y = 4*x
    y1 = [0]
    for k in range(1, 20, 2):
        y1.append(64/((k*(pi))**3)*np.exp((-9/4)*((pi)**2)*(k**2)*t)*np.sin((k*pi*x)/2))

    y = y - sum(y1)  

    return y

# set numerical parameters
mx = 1000    # number of gridpoints in space
mt = 10000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]

u_j = bc.backward_euler_solver_dirichlet(u_I,params,left_BC_fun, right_BC_fun, lambda x,t: 0*x*t)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()
"""

### --------------------------------------------------------------###


"""

# Problem 3:    (Homogeneous Neumann BCs)


#   u_t = kappa u_xx  0<x<L, 0<t<T
# with Neumann boundary conditions
#   dudx(0,t) = 0, dudx(L,t) = 0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=1        # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.cos(pi*x)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-(pi**2)*t)*np.cos(pi*x)
    return y


# set numerical parameters
mx = 1000    # number of gridpoints in space
mt = 10000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]

u_j = bc.backward_euler_solver_neumann(u_I,params)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()


"""
### --------------------------------------------------------------###



# Problem 4:    (Dirichelt BCs dependant on time, non-homogeneous Heat equation with source.)


#   u_t = kappa u_xx + g(x,t) 0<x<L, 0<t<T
# with dirichlet boundary conditions
#   u(0,t) = g1(t), u(L,t) = g2(t)
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5   # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.cos(pi*x) + x**2
    return y

def u_exact(x,t):
    # the exact solution
    y = x**2+4*x*t+np.exp(-t)*np.cos(pi*x)
    return y

def source(x,t):
    # Forcing/source function in given problem
    y = (pi**2-1)*np.exp(-t)*np.cos(pi*x)+4*x-2
    return y

left_BC_fun = lambda t: np.exp(-t)
right_BC_fun = lambda t: -np.exp(-t)+4*t+1


# set numerical parameters
mx = 100    # number of gridpoints in space
mt = 1000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]


u_j = bc.backward_euler_solver_dirichlet(u_I,params,left_BC_fun,right_BC_fun,source)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

"""


### ------------------------------------------------------------------------###

# Problem 5: (non-homogeneous heat equation with non-homogeneous Neumann boundary conditions)

#   u_t = kappa u_xx  + f(x,t) 0<x<L, 0<t<T
# with Neumann boundary conditions
#   dudx(0,t) = g1(t), dudx(L,t) = g2(t)
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.1      # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.cos(pi*x) + x**2
    return y


def u_exact(x,t):
    # the exact solution
    y = x**2+x*t+np.exp(-0.5*(pi**2)*t)*np.cos(pi*x)
    return y


def source(x,t):
    # Forcing/source function in given problem
    y = 0.5*(pi**2)*np.exp(-0.5*(pi**2)*t)*np.cos(pi*x)+x-2
    return y

left_BC_fun = lambda t:  t
right_BC_fun = lambda t: t+2


# set numerical parameters
mx = 100    # number of gridpoints in space
mt = 1000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]


u_j = bc.backward_euler_solver_neumann(u_I,params,left_BC_fun,right_BC_fun,source)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()


### ------------------------------------------------------------------------###


"""


















