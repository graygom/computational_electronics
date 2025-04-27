#
# TITLE: Computational Electronics
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/playlist?list=PLJtAfFg1nIX9dsWGnbgFt2dqvwXVRSVxm
#


import numpy as np
import matplotlib.pyplot as plt


#===================
# Lecture 00, 01
#
# C++
# MATLAB
# Python


#===================
# Lecture 02: eigenvalue problem
#
# pip install numpy
# np.linalg.eig()

A = np.array([[2, -1], [-1, 2]], dtype=float)
D, V = np.linalg.eig(A)

if False:
    print(A)    # matrix
    print(D)    # eigenvalues
    print(V)    # eigenvectors


#===================
# Lecture 03: 1D infinite potential well
#
# Schrodinger equation
#
# wavefunction
# boundary conditions: Dirichlet BCs
#                      0.0@left-end, 0.0@right-end
# discretization
# eigenvalue problem

length = 1e-6                   # in m
number_of_elements = 50         # ea

# fundamental constants in SI unit
h = 6.62607015e-34              # J * s
hbar = 1.05457182e-34           # m^2 * kg / s
m_e = 9.1093837e-31             # kg

# additional parameters
dx = length / number_of_elements
number_of_nodes = number_of_elements + 1

# global matrix
matrix_rows = number_of_nodes - 2      # inner nodes
matrix_cols = number_of_nodes          # left side end, inner nodes, right side end  
A = np.zeros([matrix_rows, matrix_cols], dtype=float)

# 
for each_inner_node in range(matrix_rows):
    row  = each_inner_node
    cols = np.arange(each_inner_node, each_inner_node+3)
    A[row, cols] += np.array([1, -2, 1], dtype=float)

#
A_Dirichlet_BC = A[:, 1:-1]         # left side end = right side end = 0.0

#
D, V = np.linalg.eig(A_Dirichlet_BC)

# lecture 3 debugging 
if False:
    print(A)
    print(A_Dirichlet_BC)
    print(D)
    print(V)


#===================
# Lecture 04: 1D infinite potential well
#
# pip install matplotlib
#
# np.argsort()

# sorting from ground state
states_index = np.argsort(-D)       # D = -k^2 * dx^2

# states
eigenvalue_list = []
eigenvector_list = []

for each_state_index in states_index[:4]:
    eigenvalue_list.append(-D[each_state_index])
    eigenvector_list.append(np.array([0.0]+list(V[:,each_state_index])+[0.0]))

# plot
if False:
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    for index in range(len(eigenvalue_list)):
        ax[index//2, index%2].plot(eigenvector_list[index],'o-')
        ax[index//2, index%2].set_title(eigenvalue_list[index])
        ax[index//2, index%2].grid(ls=':')
    plt.show()


#===================
# Lecture 05: 1D Laplace equation
#
# Laplace equation: div( grad( phi ) ) = 0
#
# np.linalg.solve()

length = 1e-6                   # in m
number_of_elements = 50         # ea

# Dirichlet BC
bias_left_end = 0.0
bias_right_end = 1.0

# fundamental constants in SI unit
ep0 = 8.8541878188e-12          # F / m

# additional parameters
dx = length / number_of_elements
number_of_nodes = number_of_elements + 1

# LHS_matrix, RHS_vector
matrix_rows = number_of_nodes           # left side end, inner nodes, right side end
matrix_cols = number_of_nodes           # left side end, inner nodes, right side end  

LHS_matrix = np.zeros([matrix_rows, matrix_cols], dtype=float)
RHS_vector = np.zeros([matrix_rows], dtype=float)

#
for each_node_index in range(number_of_nodes):
    if each_node_index == 0:
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_left_end             # Dirichlet BC
        
    elif each_node_index == (number_of_nodes-1):
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_right_end            # Dirichlet BC

    else:
        LHS_matrix[each_node_index, (each_node_index-1):(each_node_index+2)] = np.array([1,-2,1], dtype=float)
        RHS_vector[each_node_index] = 0.0

# solution of linear equations
V = np.linalg.solve(LHS_matrix, RHS_vector)
x = np.linspace(0.0, length, number_of_nodes)

# plot
if False:
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(x, V, 'o-')
    ax.set_xlabel('position [m]')
    ax.set_ylabel('electric potential V [V]')
    ax.set_title('solving 1D Laplace equation')
    ax.grid(ls=':')
    plt.show()
    

#===================
# Lecture 06: 1D Poisson equation without fixed charge
#
# Poisson equation: div( ep( r ) * grad( phi ) ) = - rho( r )  = 0 (no fixed charge)
#
#    V_n-1       V_n         V_n+1
#     o-----------o-----------o
#        ep_n-1        ep_n
#        rho_n-1       rho_n
#           |            |
# integrating over control volume
#
#   int_{n-0.5}^{n+0.5} { d( ep(x) * dphi / dx ) / dx } * dx
#              = ep(n+0.5) * dphi/dx_{n+0.5} - ep(n-0.5) * dphi/dx_{n-0.5}
#              = ep(n+0.5) * ( phi_n+1 - phi_n) / dx
#                       - ep(n-0.5) * ( phi_n - phi_n-1) / dx 
#              = ( ep(n+0.5) * phi_n+1 - ( ep(n+0.5) + ep(n-0.5) ) * phi_n + ep(n-0.5) * phi_n-1 ) / dx
#
#              = 0 (no fixed charge)

length = 1e-6                   # in m
number_of_elements = 50         # ea

ep1 = 11.7                      # dielectric constant 1
ep2 = 3.9                       # dielectric constant 2

bias_left_end = 0.0
bias_right_end = 1.0

# fundamental constants in SI unit
ep0 = 8.8541878188e-12          # F / m

# additional parameters
dx = length / number_of_elements
number_of_nodes = number_of_elements + 1

# dielectric constant
ep = np.zeros(number_of_elements, dtype=float)
for index in range(number_of_elements):
    if index < int(number_of_elements/2):
        ep[index] = ep1
    else:
        ep[index] = ep2

# LHS_matrix, RHS_vector
matrix_rows = number_of_nodes           # left side end, inner nodes, right side end
matrix_cols = number_of_nodes           # left side end, inner nodes, right side end  

LHS_matrix = np.zeros([matrix_rows, matrix_cols], dtype=float)
RHS_vector = np.zeros([matrix_rows], dtype=float)

#
for each_node_index in range(number_of_nodes):
    if each_node_index == 0:
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_left_end
        
    elif each_node_index == (number_of_nodes-1):
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_right_end

    else:
        LHS_matrix[each_node_index, each_node_index-1] = ep[each_node_index-1]
        LHS_matrix[each_node_index, each_node_index]   = -(ep[each_node_index-1] + ep[each_node_index])
        LHS_matrix[each_node_index, each_node_index+1] = ep[each_node_index]
        RHS_vector[each_node_index] = 0.0

# solution of linear equations
V = np.linalg.solve(LHS_matrix, RHS_vector)
x = np.linspace(0.0, length, number_of_nodes)

# plot
if False:
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(x, V, 'o-')
    ax.set_xlabel('position [m]')
    ax.set_ylabel('electric potential V [V]')
    ax.set_title('solving 1D Poisson equation (w/o fixed charge)')
    ax.grid(ls=':')
    plt.show()


#===================
# Lecture 07: 1D Poisson equation with fixed charge
#
# charges inside semiconductor: rho(r) = q*p(r) - q*n(r) + q*N_dop+(r)
#
# Poisson equation: div( ep( r ) * grad( phi ) ) = - rho( r )   (with fixed charge)
#
#    V_n-1       V_n         V_n+1
#     o-----------o-----------o
#        ep_n-1        ep_n
#        rho_n-1       rho_n
#           |            |
# integrating over control volume
#
#   int_{n-0.5}^{n+0.5} { d( ep(x) * dphi / dx ) / dx } * dx
#              = ep(n+0.5) * dphi/dx_{n+0.5} - ep(n-0.5) * dphi/dx_{n-0.5}
#              = ep(n+0.5) * ( phi_n+1 - phi_n) / dx
#                       - ep(n-0.5) * ( phi_n - phi_n-1) / dx 
#              = ( ep(n+0.5) * phi_n+1 - ( ep(n+0.5) + ep(n-0.5) ) * phi_n + ep(n-0.5) * phi_n-1 ) / dx
#
#   int_{n-0.5}^{n+0.5} { - rho( r ) } * dx
#              = -q * ( N_dop+(n-0.5) *  (x_n - x_n-1) + N_dop+(n+0.5) *  (x_n+1 - x_n) ) / 2
#
# scaling = l0 / ep0


length = 6e-9                   # in m
number_of_elements = 50         # ea

ep1 = 3.9                       # dielectric constant 1
ep2 = 11.7                      # dielectric constant 2
ep3 = 3.9                       # dielectric constant 3

rho1 = 0.0                      # fixed charge 1
rho2 = -1e24                     # fixed charge 2
rho3 = 0.0                      # fixed charge 3

bias_left_end = 0.0
bias_right_end = 0.0

# fundamental constants in SI unit
ep0 = 8.8541878188e-12              # F / m
q = 1.60217663e-19                  # C

# additional parameters
dx = length / number_of_elements
number_of_nodes = number_of_elements + 1

# dielectric constant, fixed charge
ep = np.zeros(number_of_elements, dtype=float)
rho = np.zeros(number_of_elements, dtype=float)

for index in range(number_of_elements):
    if index < 5:
        ep[index] = ep1 * ep0
        rho[index] = rho1
    elif index > (number_of_elements-5):
        ep[index] = ep3 * ep0
        rho[index] = rho3
    else:
        ep[index] = ep2 * ep0
        rho[index] = rho2

# LHS_matrix, RHS_vector
matrix_rows = number_of_nodes           # left side end, inner nodes, right side end
matrix_cols = number_of_nodes           # left side end, inner nodes, right side end  

LHS_matrix = np.zeros([matrix_rows, matrix_cols], dtype=float)
RHS_vector = np.zeros([matrix_rows], dtype=float)

#
scaling = dx / ep0

for each_node_index in range(number_of_nodes):
    if each_node_index == 0:
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_left_end
        
    elif each_node_index == (number_of_nodes-1):
        LHS_matrix[each_node_index, each_node_index] = 1.0
        RHS_vector[each_node_index] = bias_right_end

    else:
        LHS_matrix[each_node_index, each_node_index-1] = ep[each_node_index-1] / dx * scaling
        LHS_matrix[each_node_index, each_node_index]   = -(ep[each_node_index-1] + ep[each_node_index]) / dx * scaling
        LHS_matrix[each_node_index, each_node_index+1] = ep[each_node_index] / dx * scaling
        RHS_vector[each_node_index] = -q * ( rho[each_node_index-1] + rho[each_node_index] ) / 2 * dx * scaling 

# solution of linear equations
V = np.linalg.solve(LHS_matrix, RHS_vector)
x = np.linspace(0.0, length, number_of_nodes)

# plot
if True:
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(x, V, 'o-')
    ax.set_xlabel('position [m]')
    ax.set_ylabel('electric potential V [V]')
    ax.set_title('solving 1D Poisson equation (w/ fixed charge)')
    ax.grid(ls=':')
    plt.show()


#===================
# Lecture 08

#===================
# Lecture 09

#===================
# Lecture 10

#===================
# Lecture 11

#===================
# Lecture 12

#===================
# Lecture 13

#===================
# Lecture 14

#===================
# Lecture 15

#===================
# Lecture 16

#===================
# Lecture 17

#===================
# Lecture 18

#===================
# Lecture 19

#===================
# Lecture 20
