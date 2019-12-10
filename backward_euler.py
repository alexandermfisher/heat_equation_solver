
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

def generate_tridiag_matrix(dim, entries):


    main_diag  = np.zeros(dim-1)
    lower_diag = np.zeros(dim-2)
    upper_diag = np.zeros(dim-2)

    # Insert main diaginals
    main_diag[:] =   entries[0]                   # 1 + 2*lmbda
    lower_diag[:] =  entries[1]                   #-lmbda
    upper_diag[:] =  entries[1]                   #-lmbda

    A = diags(diagonals=[main_diag, lower_diag, upper_diag],offsets=[0, -1, 1], shape=(dim-1, dim-1),format="csr")


    return A



def backward_euler_solver_dirichlet(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t):

    import time;  t0 = time.clock()  # for measuring the CPU time

    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = left_BC_fun(t);   right_BC = right_BC_fun(t)
    

    # Data Structure for linear system using sparse matrices:
    A = generate_tridiag_matrix(mx,[1+2*lmbda,-lmbda])
    

    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):    u_j[i] = u_I(x[i])
    u_j[0] = left_BC[0];    u_j[-1] = right_BC[0]
    

    for n in range(1, mt+1):
        b = u_j[1:-1] + deltat*f(x[1:-1], t[n])
        b[0] = b[0] + lmbda*left_BC[n]
        b[-1] = b[-1] + lmbda*right_BC[n]
    
        u_jn[1:-1] = spsolve(A,b)

        # Update u_j
        u_j[1:-1] = u_jn[1:-1]
        u_j[0] = left_BC[n+1] 
        u_j[-1] = right_BC[n+1]


    t1 = time.clock()
    run_time = float(t1-t0)

    return u_j, run_time



def backward_euler_solver_neumann(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t):

    import time;  t0 = time.clock()  # for measuring the CPU time


    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = list(map(left_BC_fun,t));   right_BC = list(map(right_BC_fun,t))
    
    


    # Data Structure for linear system using sparse matrices:

    A = generate_tridiag_matrix(mx+2,[1+2*lmbda,-lmbda])
    A[0,1] = -2*lmbda; A[-1,-2] = -2*lmbda;

    
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    for n in range(1, mt+1):
        b = u_j + deltat*f(x[1:-1], t[n])
        b[0] = b[0] - 2*lmbda*deltax*left_BC[n]
        b[-1] = b[-1] + 2*lmbda*deltax*right_BC[n]
    
        u_jn = spsolve(A,b)

        # Update u_j
        u_j = u_jn
        


    t1 = time.clock()
    run_time = float(t1-t0)

    return u_j, run_time





















