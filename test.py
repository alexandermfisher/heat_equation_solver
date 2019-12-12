"""
test.py:

Test script that uses functions in heat_equation_function_lib.py to solve various types of diffusion problems.

"""
import heat_equation_function_lib as solver
import numpy as np
import pylab as pl
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import pi



###-----------------------------------------------------------------###

### Problems: various diffusion equations with differing boundry conditions. See link for more,
### http://www.iaeng.org/publication/IMECS2014/IMECS2014_pp535-539.pdf


def plot(u_j,L,mx,mt,T,u_exact):
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    pl.plot(x,u_j,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,'+ str(T) +')')
    pl.legend(loc='upper right')
    pl.show()


def problem_1(method = "backward"):
    """
    Problem 1:  Homogeneous Dirichelt BCs
    
    u_t = kappa u_xx  0<x<L, 0<t<T
    with zero-temperature boundary conditions, u=0 at x=0,L, t>0
    and prescribed initial temperature u=u_I(x) 0<=x<=L,t=0

    """
    # set problem parameters/functions
    kappa = 1.0   # diffusion constant
    L=1.0         # length of spatial domain
    T=0.6        # total time to solve for

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

    if method == "backward":        u_j = solver.backward_euler_solver_dirichlet(u_I,params)
    if method == "crank_nicolson":  u_j = solver.crank_nicolson_solver_dirichlet(u_I,params) 

    ### plot the final result and exact solution ###
    plot(u_j,L,mx,mt,T,u_exact)


def problem_2(method = "backward"):
    """
    Problem 3:  Non-Homogeneous Dirichelt BCs
    
    u_t = kappa u_xx  0<x<L, 0<t<T
    with temperature boundary conditions, u(0,t) = constant_1, u(L,t) = constant_2
    and prescribed initial temperature u=u_I(x) 0<=x<=L,t=0
    """

    # set problem parameters/functions
    kappa = 9.0   # diffusion constant
    L=2.0         # length of spatial domain
    T=1        # total time to solve for

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
    mx = 100    # number of gridpoints in space
    mt = 10000  # number of gridpoints in time

    params = [kappa,L,T,mx,mt]

    if method == "backward":        u_j = solver.backward_euler_solver_dirichlet(u_I,params,left_BC_fun,right_BC_fun)
    if method == "crank_nicolson":  u_j = solver.crank_nicolson_solver_dirichlet(u_I,params,left_BC_fun,right_BC_fun) 

    plot(u_j,L,mx,mt,T,u_exact)


def problem_3(method = "backward"):
    """
    Problem 3:  Homogeneous Neumann BCs
    
    u_t = kappa u_xx  0<x<L, 0<t<T
    with with Neumann boundary conditions, dudx(0,t) = 0, dudx(L,t) = 0
    and prescribed initial temperature u=u_I(x) 0<=x<=L,t=0
    """
    # set problem parameters/functions
    kappa = 1.0   # diffusion constant
    L=1.0         # length of spatial domain
    T=0.01      # total time to solve for

    def u_I(x):
        # initial temperature distribution
        y = np.cos(pi*x)
        return y

    def u_exact(x,t):
        # the exact solution
        y = np.exp(-(pi**2)*t)*np.cos(pi*x)
        return y


    # set numerical parameters
    mx = 100   # number of gridpoints in space
    mt = 10000  # number of gridpoints in time

    params = [kappa,L,T,mx,mt]

    if method == "backward":        u_j = solver.backward_euler_solver_neumann(u_I,params)
    if method == "crank_nicolson":  u_j = solver.crank_nicolson_solver_neumann(u_I,params) 

    plot(u_j,L,mx,mt,T,u_exact)


def problem_4(method = "backward"):
    """
    Problem 4:  Dirichelt BCs dependant on time, non-homogeneous Heat equation with source.)
    
    u_t = kappa u_xx + g(x,t) 0<x<L, 0<t<T
    with dirichlet boundary conditions, u(0,t) = g1(t), u(L,t) = g2(t)
    and prescribed initial temperature u=u_I(x) 0<=x<=L,t=0
    """
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
    mx = 100   # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    params = [kappa,L,T,mx,mt]

    if method == "backward":        u_j = solver.backward_euler_solver_dirichlet(u_I,params,left_BC_fun,right_BC_fun,source)
    if method == "crank_nicolson":  u_j = solver.crank_nicolson_solver_dirichlet(u_I,params,left_BC_fun,right_BC_fun,source) 

    plot(u_j,L,mx,mt,T,u_exact)


def problem_5(method = "backward"):
    """
    Problem 5:  non-homogeneous heat equation with non-homogeneous Neumann boundary conditions
    with Neumann boundary conditions dudx(0,t) = g1(t), dudx(L,t) = g2(t)
    and prescribed initial temperature u=u_I(x) 0<=x<=L,t=0
    """

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

    if method == "backward":        u_j = solver.backward_euler_solver_neumann(u_I,params,left_BC_fun,right_BC_fun,source)
    if method == "crank_nicolson":  u_j = solver.crank_nicolson_solver_neumann(u_I,params,left_BC_fun,right_BC_fun,source) 

    plot(u_j,L,mx,mt,T,u_exact)


###-----------------------------------------------------------------###

### Running Tests:

problem_1("backward"); problem_1("crank_nicolson")

problem_2("backward"); problem_2("crank_nicolson")

problem_3("backward"); problem_3("crank_nicolson")

problem_4("backward"); problem_4("crank_nicolson")

problem_5("backward"); problem_5("crank_nicolson")

















