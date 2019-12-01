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
mx = 5000    # number of gridpoints in space
mt = 10000   # number of gridpoints in time

x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)



# Data Structure for linear system using sparse matrices:

# Matrix A
main_diag_A  = np.zeros(mx+1)
lower_diag_A = np.zeros(mx)
upper_diag_A = np.zeros(mx)

main_diag_A[:] = 1 + lmbda
lower_diag_A[:] = -lmbda/2
upper_diag_A[:] = -lmbda/2

# Insert boundary conditions
main_diag_A[0] = 0; main_diag_A[mx] = 0


A = diags(diagonals=[main_diag_A, lower_diag_A, upper_diag_A],offsets=[0, -1, 1], shape=(mx+1, mx+1),format="csr")


# Matrix B
main_diag_B = np.zeros(mx+1)
lower_diag_B = np.zeros(mx)
upper_diag_B = np.zeros(mx)

main_diag_B[:] = 1 - lmbda
lower_diag_B[:] = lmbda/2
upper_diag_B[:] = lmbda/2


# Insert boundary conditions
main_diag_B[0] = 0; main_diag_B[mx] = 0


B = diags(diagonals=[main_diag_B, lower_diag_B, upper_diag_B],offsets=[0, -1, 1], shape=(mx+1, mx+1),format="csr")

"""
print(A.todense())
print(B.todense())
"""



# set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jn = np.zeros(x.size)      # u at next time step


# Set initial condition
for i in range(0, mx+1):
    u_j[i] = u_I(x[i])

for n in range(0, mt):
    b = u_j
    b[0] = b[-1] = 0.00000000
    u_jn[:] = spsolve(A, B.dot(b))

    # Update u_j
    u_j[:] = u_jn[:]



# plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()
