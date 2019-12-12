
"""
heat_equation_function_lib.py:

This function library contians implementations of finite differnce schemes for numerically solving the heat/diffusion equation in 1D.
The schemese included are backwards-euler method and crank-nicolson method. Below is a list of included functions; see main functions below for more detail.

-   initilise_numerical_variables(params)

-   initlise_solution_variables(init_con, x)

-   generate_tridiag_matrix(dim, entries)

-   backward_euler_solver_dirichlet(init_con, params, left_BC_fun, right_BC_fun, source_fun)

-   backward_euler_solver_neumann(init_con, params, left_BC_fun, right_BC_fun, source_fun)

-   crank_nicolson_solver_dirichlet(init_con, params, left_BC_fun, right_BC_fun, source_fun)

-   crank_nicolson_solver_neumann(init_con, params, left_BC_fun, right_BC_fun, source_fun)

"""
# Standard imports
import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve



###------------------------------------ Function_Library -------------------------------------------###

 
# peliminary functions to reduce code in the main 'numerical scheme' functions to follow

def initilise_numerical_variables(params):
    
    kappa, L, T, mx, mt = params            # unpack parameters

    x = np.linspace(0, L, mx+1)             # mesh points in space
    t = np.linspace(0, T, mt+1)             # mesh points in time
    deltax = x[1] - x[0]                    # gridspacing in x
    deltat = t[1] - t[0]                    # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)        # mesh fourier number

    return x, t, deltax, deltat, lmbda

def initlise_solution_variables(init_con, x):

    u_j = np.zeros(x.size)                  # u at current time step
    u_jn = np.zeros(x.size)                 # u at next time step
    
    for i in range(0, x.size):              # solve initial conditions and store in u_j
        u_j[i] = init_con(x[i])

    return u_j, u_jn

def generate_tridiag_matrix(dim, entries):

    main_diag  = np.zeros(dim-1)            # initilise tridiagonal matrix with specified dim. 
    lower_diag = np.zeros(dim-2)
    upper_diag = np.zeros(dim-2)

                                            
    main_diag[:] =   entries[0]             # insert main diaginals 
    lower_diag[:] =  entries[1]             
    upper_diag[:] =  entries[1]                   

    # define sparse matrix with given dim and entries
    A = diags(diagonals=[main_diag, lower_diag, upper_diag],offsets=[0, -1, 1], shape=(dim-1, dim-1),format="csr")


    return A




###########################
###  Numerical Schemes  ###
###########################



#### Backward Euler (Implicit Euler) Method with Dirichelt Boundry Conditions ####

def backward_euler_solver_dirichlet(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t):
    """

    This function performs backward euler method to numerically solve heat equations with dirichlet boundry conditions. 


    Parameters
    ----------

            init_con:           callable
                                prescribed initial temperature distibution u(x,0) = u_I(x) where x: [0,L].

            
            params:             ndarray,shape(5,)
                                List of problem parameters. Paramaters must be in the correct 
                                order as in [kappa,L,T,mx,mt], which are diffusion constant,
                                length of spatial domain, total time to solve to, number of gridpoints in space,
                                and number of gridpoints in time respectively. 


            left_BC_fun:        callable, optional
                                prescribed 'left' dirichlet boundary conditions u(0,t) = g1(t). Default set to homogenous boundry 
                                condition, i.e. u(0,t) = 0. 

            right_BC_fun:       callable, optional
                                prescribed 'right' dirichlet boundary conditions u(L,t) = g2(t). Default set to homogenous boundry 
                                condition, i.e. u(L,t) = 0. 

            source_fun:         callable, optional
                                prescribed heat source or forcing function, source = f(x,t). Default set to have no heat source 
                                for solving homogeneous heat equations.  

    Returns
    -------     
            u_j:                ndarray,shape(mx+1,)
                                solution vector solver for time T, i.e u(x,T) = g(x) for x: [0,L]. 

                        
    ---------------------------------------------------------------------------------------------------
    """


    # unpack parmaaters and initilise numerical variables and boundry conditions. 
    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = left_BC_fun(t) ;  right_BC = right_BC_fun(t)
    

    # Data Structure for linear system using sparse matrices:
    A = generate_tridiag_matrix(mx,[1+2*lmbda,-lmbda])
    
    # Initlise soulution variables and insert dirichlet boundry conditions.
    u_j, u_jn = initlise_solution_variables(u_I,x)
    u_j[0] = left_BC[0];    u_j[-1] = right_BC[0]
    

    # solve linear system by time stepping and update current u_j = (x,t) 
    for n in range(1, mt+1):
        b = u_j[1:-1] + deltat*f(x[1:-1], t[n])     # currnet u_j + forcing terms
        b[0] = b[0] + lmbda*left_BC[n]              # adding dirichlet conditions
        b[-1] = b[-1] + lmbda*right_BC[n]
    
        u_jn[1:-1] = spsolve(A,b)

        # Update u_j
        u_j[1:-1] = u_jn[1:-1]
        u_j[0] = left_BC[n] 
        u_j[-1] = right_BC[n]


    return u_j


#### Backward Euler Method with Neumann Boundry Conditions ####

def backward_euler_solver_neumann(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t+0*x):
    """

    This function performs backward euler method to numerically solve heat equations with neumann boundry conditions. 


    Parameters
    ----------

            init_con:           callable
                                prescribed initial temperature distibution u(x,0) = u_I(x) where x: [0,L].

            
            params:             ndarray,shape(5,)
                                List of problem parameters. Paramaters must be in the correct 
                                order as in [kappa,L,T,mx,mt], which are diffusion constant,
                                length of spatial domain, total time to solve to, number of gridpoints in space,
                                and number of gridpoints in time respectively. 


            left_BC_fun:        callable, optional
                                prescribed 'left' neumann boundary conditions dudx(0,t) = g1(t). Default set to homogenous boundry 
                                condition, i.e. dudx(0,t) = 0. 

            right_BC_fun:       callable, optional
                                prescribed 'right' neumann boundary conditions dudx(L,t) = g2(t). Default set to homogenous boundry 
                                condition, i.e. dudx(L,t) = 0. 

            source_fun:         callable, optional
                                prescribed heat source or forcing function, source = f(x,t). Default set to have no heat source 
                                for solving homogeneous heat equations.  

    Returns
    -------     
            u_j:                ndarray,shape(mx+1,)
                                solution vector solver for time T, i.e u(x,T) = g(x) for x: [0,L]. 

                        
    ---------------------------------------------------------------------------------------------------
    """

    # unpack parmaaters and initilise numerical variables and boundry conditions. 
    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = left_BC_fun(t);   right_BC = right_BC_fun(t)
    

    # Data Structure for linear system using sparse matrices:
    A = generate_tridiag_matrix(mx+2,[1+2*lmbda,-lmbda])
    A[0,1] = -2*lmbda; A[-1,-2] = -2*lmbda;

    # Iniitlise soulution variables
    u_j, u_jn = initlise_solution_variables(u_I,x)


    # solve linear system by time stepping and update current u_j = (x,t) 
    for n in range(1, mt+1):
        b = u_j + deltat*f(x,t[n])
        b[0] = b[0] - 2*lmbda*deltax*left_BC[n]
        b[-1] = b[-1] + 2*lmbda*deltax*right_BC[n]
    
        u_jn = spsolve(A,b)

        # Update u_j
        u_j = u_jn

    return u_j


#### Crank-Nicolson Method with Dirichlet Boundry Conditions ####


def crank_nicolson_solver_dirichlet(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t):

    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = left_BC_fun(t);   right_BC = right_BC_fun(t)


    # Data Structure for linear system using sparse matrices:
    A = generate_tridiag_matrix(mx,[2+2*lmbda,-lmbda])
    B = generate_tridiag_matrix(mx,[2-2*lmbda,lmbda])
   
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    

    # Initlise soulution variables and insert dirichlet boundry conditions.
    u_j, u_jn = initlise_solution_variables(u_I,x)
    u_j[0] = left_BC[0];    u_j[-1] = right_BC[0]
    

    # solve linear system by time stepping and update current u_j = (x,t) 
    for n in range(1, mt+1):
        b = u_j[1:-1] + deltat*f(x[1:-1], t[n])
        C = B.dot(b)
        C[0] = C[0] + lmbda*(left_BC[n-1]+left_BC[n])
        C[-1] = C[-1] + lmbda*(right_BC[n-1]+right_BC[n])
        
        u_jn[1:-1] = spsolve(A, C)

        # Update u_j
        u_j[1:-1] = u_jn[1:-1]
        u_j[0] = left_BC[n] 
        u_j[-1] = right_BC[n]


    return u_j


#### Crank-Nicolson Method with Neumann Boundry Conditions ####

def crank_nicolson_solver_neumann(init_con, params, left_BC_fun = lambda t: 0*t, right_BC_fun = lambda t: 0*t, source_fun = lambda x,t: 0*t):

    kappa, L, T, mx, mt = params
    u_I = init_con; f = source_fun
    x, t, deltax, deltat, lmbda = initilise_numerical_variables(params)
    left_BC = left_BC_fun(t);   right_BC = right_BC_fun(t)

    # Data Structure for linear system using sparse matrices:

    A = generate_tridiag_matrix(mx+2,[1+lmbda,-0.5*lmbda])
    A[0,1] = -lmbda; A[-1,-2] = -lmbda;

    B = generate_tridiag_matrix(mx+2,[1-lmbda,0.5*lmbda])
    B[0,1] = lmbda; B[-1,-2] = lmbda;
    

   
   
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jn = np.zeros(x.size)      # u at next time step

    
    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    for n in range(1, mt+1):
        b = u_j  #+ deltat*f(x[1:-1],t[n])
        C = B.dot(b)
        C[0] = C[0] #- 2*lmbda*deltax*(left_BC[n-1]+left_BC[n])
        C[-1] = C[-1] #+ 2*lmbda*deltax*(right_BC[n-1]+right_BC[n])
    
        u_jn = spsolve(A,C)
        # Update u_j
        u_j = u_jn


        return u_j


