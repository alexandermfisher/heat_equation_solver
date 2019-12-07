
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

    main_diag  = np.zeros(mx-1)
    lower_diag = np.zeros(mx-2)
    upper_diag = np.zeros(mx-2)

    # Insert main diaginals
    main_diag[:] = 1 + 2*lmbda
    lower_diag[:] = -lmbda
    upper_diag[:] = -lmbda

    A = diags(diagonals=[main_diag, lower_diag, upper_diag],offsets=[0, -1, 1], shape=(mx-1, mx-1),format="csr")
    

    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step


    # Set initial condition
    u_j = u_I(x)
    u_j[0] = u_j[-1] = 0


    for n in range(0, mt+1):
        b = u_j[1:-1]
        u_jn[1:-1] = spsolve(A,b)

        # Update u_j
        u_j[1:-1] = u_jn[1:-1]
        u_j[0] = u_j[-1] = 0


    return u_j



def backward_euler_solver_dirichlet_BC(init_con,left_BC_fun,right_BC_fun,params):

    kappa, L, T, mx, mt = params
    u_I = init_con
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)

    left_BC = left_BC_fun(t);   right_BC = right_BC_fun(t)
    left_BC = np.append(left_BC,[left_BC_fun(T+deltat)]);  right_BC = np.append(right_BC,[right_BC_fun(T+deltat)])


    # Data Structure for linear system using sparse matrices:

    main_diag  = np.zeros(mx-1)
    lower_diag = np.zeros(mx-2)
    upper_diag = np.zeros(mx-2)

    # Insert main diaginals
    main_diag[:] = 1 + 2*lmbda
    lower_diag[:] = -lmbda
    upper_diag[:] = -lmbda

    A = diags(diagonals=[main_diag, lower_diag, upper_diag],offsets=[0, -1, 1], shape=(mx-1, mx-1),format="csr")
    

    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    # Set initial condition
    u_j = u_I(x)
    u_j[0] = left_BC[0];    u_j[-1] = right_BC[0]
    

    for n in range(0, mt+1):
        b = u_j[1:-1]
        b[0] = b[0] + lmbda*left_BC[n+1]
        b[-1] = b[-1] + lmbda*right_BC[n+1]
    
        u_jn[1:-1] = spsolve(A,b)

        # Update u_j
        u_j[1:-1] = u_jn[1:-1]
        u_j[0] = left_BC[n+1] 
        u_j[-1] = right_BC[n+1]

    return u_j





















