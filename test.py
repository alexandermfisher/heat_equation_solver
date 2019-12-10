### Test File

import backward_euler as bc
import crank_nicolson as cn
import numpy as np
import pylab as pl
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import exp, pi, cos, sin
import math



###------------------------------------------------------###
"""

# Problem 1: 

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
left_BC_fun = lambda t:  0*t + 0
right_BC_fun = lambda t: 0*t + 0

#### Solve using either crank_nicolson or bacward euler methods ####

#u_j = cn.crank_nicolson_solver_homogeneous_BC(u_I, params)
#u_j, run_time = bc.backward_euler_solver_dirichlet(u_I,params)
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
"""

# Problem 2"

#   u_t = kappa u_xx  0<x<L, 0<t<T
# with temperature boundary conditions
#   u(0,t) = constant_1, u(L,t) = constant_2
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

# set problem parameters/functions
kappa = 9.0   # diffusion constant
L=2.0         # length of spatial domain
T=0.5        # total time to solve for

left_BC_fun = lambda t:  0*t + 0
right_BC_fun = lambda t: 0*t + 8

def u_I(x):
    # initial temperature distribution
    y = 2*x**2
    return y

def u_exact(x,t):

    y = 4*x
    y1 = []
    for k in range(1, 20, 2):
        y1.append(64/((k*(np.pi))**3)*math.exp((-9/4)*((np.pi)**2)*(k**2)*t)*np.sin((k*np.pi*x)/2))

    y = y - sum(y1)  

    return y


# set numerical parameters
mx = 100    # number of gridpoints in space
mt = 1000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]


#### Solve using either crank_nicolson or bacward euler methods ####


u_j, time = bc.backward_euler_solver(u_I,params,left_BC_fun, right_BC_fun, lambda x,t: -20*x*t)


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

print(float(time))




"""
### --------------------------------------------------------------###

# Problem 3:

#Backward method to solve 1D reaction-diffusion equation:
#        u_t = k * u_xx
    
#with Neumann boundary conditions 
#at x=0: u_x(0,t) = 0 = sin(2*np.pi)
#at x=L: u_x(L,t) = 0 = sin(2*np.pi)

#with L = 1 and initial conditions:
#u(x,0) = (1.0/2.0)+ np.cos(2.0*np.pi*x) - (1.0/2.0)*np.cos(3*np.pi*x)

#u_x(x,t) = (-4.0*(np.pi**2))np.exp(-4.0*(np.pi**2)*t)*np.cos(2.0*np.pi*x) + 
#            (9.0/2.0)*(np.pi**2)*np.exp(-9.0*(np.pi**2)*t)*np.cos(3*np.pi*x))


# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=5        # total time to solve for

left_BC_fun = lambda x:  sin(2*pi*x)
right_BC_fun = lambda x: sin(2*pi*x)


def u_I(x):
    y = (1.0/2.0)+ cos(2.0*pi*x) - (1.0/2.0)*cos(3*pi*x)
    return y

def u_exact(x):
    t = T
    y = (-4.0*(pi**2))*exp(-4.0*(pi**2)*t)*cos(2.0*pi*x)+(9.0/2.0)*(pi**2)*exp(-9.0*(pi**2)*t)*cos(3*pi*x)
    return y

# set numerical parameters
mx = 100    # number of gridpoints in space
mt = 1000  # number of gridpoints in time

params = [kappa,L,T,mx,mt]

#### Solve using either crank_nicolson or bacward euler methods ####


u_j, time = bc.backward_euler_solver_neumann(u_I,params)


### plot the final result and exact solution ###

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
sol = list(map(u_exact ,xx))
print(sol)
pl.plot(xx,sol,'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

print(float(time))







### --------------------------------------------------------------###






