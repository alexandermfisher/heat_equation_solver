
import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve




def initilise_numerical_variables(params):
    
    kappa, L, T, mx, mt = params    # unpack parameters

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    return x, t, deltax, deltat, lmbda

def backward_euler_solver_homogenous_BC(init_con,params): 

    kappa, L, T, mx, mt = params
    u_I = init_con
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)



    # Data Structure for linear system using sparse matrices:

    main_diag  = np.zeros(mx+1)
    lower_diag = np.zeros(mx)
    upper_diag = np.zeros(mx)

    # Insert main diaginals
    main_diag[:] = 1 + 2*lmbda
    lower_diag[:] = -lmbda
    upper_diag[:] = -lmbda

    # Insert boundary conditions coeffcients
    main_diag[0] = 1; main_diag[-1] = 1
    upper_diag[0] = 0 ; lower_diag[-1] = 0



    A = diags(diagonals=[main_diag, lower_diag, upper_diag],offsets=[0, -1, 1], shape=(mx+1, mx+1),format="csr")
    




    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step


    # Set initial condition
    u_j = u_I(x)
    u_j[0] = u_j[-1] = 0



    for n in range(0, mt+1):
        b = u_j
        u_jn[:] = spsolve(A,b)

        # Update u_j
        u_j[:] = u_jn[:]
        u_j[0] = u_j[-1] = 0


    return u_j
























