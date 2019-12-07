# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0


import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve



### Iniitilise x-linspace and time variables ...


def initilise_numerical_variables(params):
    
    kappa, L, T, mx, mt = params    # unpack parameters

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    return x, t, deltax, deltat, lmbda




def crank_nicolson_solver_homogenous_BC(init_con, params):

    kappa, L, T, mx, mt = params
    u_I = init_con
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)


    # Data Structure for linear system using sparse matrices:

    # Matrix A
    main_diag_A  = np.zeros(mx+1)
    lower_diag_A = np.zeros(mx)
    upper_diag_A = np.zeros(mx)

    main_diag_A[:] = 1 + lmbda
    lower_diag_A[:] = -lmbda/2
    upper_diag_A[:] = -lmbda/2

    # Insert boundary conditions
    main_diag_A[0] = 1; main_diag_A[-1] = 1
    upper_diag_A[0] = 0 ; lower_diag_A[-1] = 0


    A = diags(diagonals=[main_diag_A, lower_diag_A, upper_diag_A],offsets=[0, -1, 1], shape=(mx+1, mx+1),format="csr")


    # Matrix B
    main_diag_B = np.zeros(mx+1)
    lower_diag_B = np.zeros(mx)
    upper_diag_B = np.zeros(mx)

    main_diag_B[:] = 1 - lmbda
    lower_diag_B[:] = lmbda/2
    upper_diag_B[:] = lmbda/2


    # Insert boundary conditions coefficients
    main_diag_B[0] = 1; main_diag_B[mx] = 1
    upper_diag_B[0] = 0 ; lower_diag_B[-1] = 0


    B = diags(diagonals=[main_diag_B, lower_diag_B, upper_diag_B],offsets=[0, -1, 1], shape=(mx+1, mx+1),format="csr")


    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    # Set initial condition  
    u_j = u_I(x)
    u_j[0] = u_j[-1] = 0


    for n in range(0, mt):
        b = u_j
        u_jn[:] = spsolve(A, B.dot(b))
    
        # Update u_j
        u_j[:] = u_jn[:]
        u_j[0] = u_j[-1] = 0

    return u_j


