#
# TITLE: Computational Electronics
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/playlist?list=PLJtAfFg1nIX9dsWGnbgFt2dqvwXVRSVxm
#


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


#
# CLASS: Fundamental Constants
#

class FC:
    def __init__(self):
        #
        self.q = 1.602192e-19           # in C
        self.kb = 1.380662e-23          # in J/K
        self.ep0 = 8.854187817e-12      # in F/m
        self.mu0 = 4.0e-12 * np.pi      # in H/m
        self.h = 6.62617e-34            # in J/s
        self.hb = self.h / (2*np.pi)    # in J/s
        self.m0 = 9.109534e-31          # in Kg

        # intrinsic charge density
        self.ni = 1e16                  # in m^-3


#
# CLASS: NEWTON METHOD
#

class NEWTON:
    def __init__(self):
        #
        self.fc = FC()
        # variable
        self.phi = sp.symbols('phi')
        self.f = sp.Function('f')(self.phi)
        # derivative
        self.df_dphi = self.f.diff(self.phi)


    def set_expression(self, op_temp, dopant_density):
        # user input
        self.V_T = self.fc.kb * op_temp / self.fc.q
        # user input
        self.expr  = self.fc.ni * sp.exp(  self.phi / self.V_T )
        self.expr -= self.fc.ni * sp.exp( -self.phi / self.V_T )
        self.expr -= dopant_density
        
        # derivative
        self.dexpr_dphi = self.df_dphi.subs(self.f, self.expr).doit()
        # delta phi
        self.delta_phi = -self.expr / self.dexpr_dphi


    def newton_method(self, phi_old):
        #
        vals = {self.phi: phi_old}
        #
        eval_delta_phi = self.delta_phi.evalf(subs=vals)

        #
        return (phi_old + eval_delta_phi)


#
# CLASS: NEWTON-RAPHSON METHOD
#

class NEWTON_RAPHSON:

    def __init__(self):
        # fundamental constants
        self.fc = FC()
        
        # variables
        self.phi1  = sp.symbols('phi1')
        self.phi2  = sp.symbols('phi2')
        self.phi3  = sp.symbols('phi3')
        
        # functions
        self.f1 = sp.Function('f1')(self.phi1, self.phi2, self.phi3)
        self.f2 = sp.Function('f2')(self.phi1, self.phi2, self.phi3)
        self.f3 = sp.Function('f3')(self.phi1, self.phi2, self.phi3)
        
        # derivative
        self.df1_dphi1 = self.f1.diff(self.phi1)
        self.df1_dphi2 = self.f1.diff(self.phi2)
        self.df1_dphi3 = self.f1.diff(self.phi3)
        self.df2_dphi1 = self.f2.diff(self.phi1)
        self.df2_dphi2 = self.f2.diff(self.phi2)
        self.df2_dphi3 = self.f2.diff(self.phi3)
        self.df3_dphi1 = self.f3.diff(self.phi1)
        self.df3_dphi2 = self.f3.diff(self.phi2)
        self.df3_dphi3 = self.f3.diff(self.phi3)


    def set_expression(self):
        # user input
        self.expr1  = -2*self.phi1 - sp.exp(self.phi1) + self.phi2
        self.expr2  = self.phi1 - sp.exp(self.phi2) - 2*self.phi2  + self.phi3
        self.expr3  = self.phi2 - sp.exp(self.phi3) - 2*self.phi3  + 4
        
        # derivative
        self.dexpr1_dphi1 = self.df1_dphi1.subs(self.f1, self.expr1).doit()
        self.dexpr1_dphi2 = self.df1_dphi2.subs(self.f1, self.expr1).doit()
        self.dexpr1_dphi3 = self.df1_dphi3.subs(self.f1, self.expr1).doit()
        self.dexpr2_dphi1 = self.df2_dphi1.subs(self.f2, self.expr2).doit()
        self.dexpr2_dphi2 = self.df2_dphi2.subs(self.f2, self.expr2).doit()
        self.dexpr2_dphi3 = self.df2_dphi3.subs(self.f2, self.expr2).doit()
        self.dexpr3_dphi1 = self.df3_dphi1.subs(self.f3, self.expr3).doit()
        self.dexpr3_dphi2 = self.df3_dphi2.subs(self.f3, self.expr3).doit()
        self.dexpr3_dphi3 = self.df3_dphi3.subs(self.f3, self.expr3).doit()

        # delta phi1
        self.dexpr1 = self.dexpr1_dphi1 
        

    def newton_raphson_method(self, phi_old):
        #
        vals = {self.phi1: phi_old[0], self.phi2: phi_old[1], self.phi3: phi_old[2]}

        # Jacobian matrix 1
        dexpr1_dphi1 = float(self.dexpr1_dphi1.evalf(subs=vals))
        dexpr1_dphi2 = float(self.dexpr1_dphi2.evalf(subs=vals))
        dexpr1_dphi3 = float(self.dexpr1_dphi3.evalf(subs=vals))
        dexpr2_dphi1 = float(self.dexpr2_dphi1.evalf(subs=vals))
        dexpr2_dphi2 = float(self.dexpr2_dphi2.evalf(subs=vals))
        dexpr2_dphi3 = float(self.dexpr2_dphi3.evalf(subs=vals))
        dexpr3_dphi1 = float(self.dexpr3_dphi1.evalf(subs=vals))
        dexpr3_dphi2 = float(self.dexpr3_dphi2.evalf(subs=vals))
        dexpr3_dphi3 = float(self.dexpr3_dphi3.evalf(subs=vals))

        # Jacobian matrix 2
        J = np.array([ [dexpr1_dphi1, dexpr1_dphi2, dexpr1_dphi3],
                       [dexpr2_dphi1, dexpr2_dphi2, dexpr2_dphi3],
                       [dexpr3_dphi1, dexpr3_dphi2, dexpr3_dphi3] ])

        # LHS vector 1
        expr1 = -float(self.expr1.evalf(subs=vals))
        expr2 = -float(self.expr2.evalf(subs=vals))
        expr3 = -float(self.expr3.evalf(subs=vals))

        # LHS vector 2
        R = np.array([ expr1, expr2, expr3 ])

        # delta phi
        delta_phi = np.linalg.solve(J, R)
        
        #
        return phi_old + delta_phi


#
# CLASS: Nonlinear Poisson equation
#
# ▽·( -ε(r) ▽Ψ(r) ) = q ρ(r) + q N_dop(r) - q n(r) + q p(r)
#
# ▽·( -ε(r) ▽Ψ(r) ) = q ρ(r) + q N_dop(r) - q N_int exp( q Ψ(r) / (k_b T) ) + q N_int exp( -q Ψ(r) / (k_b T) ) 
#
# f = ▽·( ε(r) ▽Ψ(r) ) q ρ(r) + q N_dop(r) - q N_int exp( q Ψ(r) / (k_b T) ) + q N_int exp( -q Ψ(r) / (k_b T) ) 
#
#

class N_POISSON_EQ:

    def __init__(self):
        # fundamental constants
        self.fc = FC()

        # variables: constant
        self.T = sp.symbols('T')
        self.phi_th = self.fc.kb * self.T / self.fc.q

        # variables: nodes
        self.phi_im10 = sp.symbols('phi_{i-1}')
        self.phi_i    = sp.symbols('phi_{i}')
        self.phi_ip10 = sp.symbols('phi_{i+1}')
        
        # variables: elements 1
        self.dx_im05 = sp.symbols('dx_{i-0.5}')
        self.dx_ip05 = sp.symbols('dx_{i+0.5}')

        # variables: elements 2
        self.ep_im05  = sp.symbols('ep_{i-0.5}')
        self.ep_ip05  = sp.symbols('ep_{i+0.5}')
        self.rho_im05 = sp.symbols('rho_{i-0.5}')
        self.rho_ip05 = sp.symbols('rho_{i+0.5]')
        self.dp_im05  = sp.symbols('dp_{i-0.5}')
        self.dp_ip05  = sp.symbols('dp_{i+0.5}')

        # electric displacement
        self.d_ip05 = self.ep_ip05 / self.dx_ip05 *( self.phi_ip10 - self.phi_i   )
        self.d_im05 = self.ep_im05 / self.dx_im05 *( self.phi_i    - self.phi_im10 )

        # charge 1: fixed charge
        self.q1_ip05 = (self.fc.q * self.rho_ip05) * (self.dx_ip05 / 2)
        self.q1_im05 = (self.fc.q * self.rho_im05) * (self.dx_im05 / 2)

        # charge 2: dopant charge
        self.q2_ip05 = (self.fc.q * self.dp_ip05) * (self.dx_ip05 / 2)
        self.q2_im05 = (self.fc.q * self.dp_im05) * (self.dx_im05 / 2)

        # charge 3: induced electrons
        self.q3_ip05 = -(self.fc.q * self.fc.ni) * (self.dx_ip05 / 2) * sp.exp( self.phi_i / self.phi_th )
        self.q3_im05 = -(self.fc.q * self.fc.ni) * (self.dx_im05 / 2) * sp.exp( self.phi_i / self.phi_th )

        # charge 4: induced holes
        self.q4_ip05 = (self.fc.q * self.fc.ni) * (self.dx_ip05 / 2) * sp.exp( -self.phi_i / self.phi_th )
        self.q4_im05 = (self.fc.q * self.fc.ni) * (self.dx_im05 / 2) * sp.exp( -self.phi_i / self.phi_th )

        # function
        self.f  = ( self.d_ip05 - self.d_im05 )
        self.f += ( self.q1_ip05 + self.q1_im05 )
        self.f += ( self.q2_ip05 + self.q2_im05 )
        self.f += ( self.q3_ip05 + self.q3_im05 )
        #self.f += ( self.q4_ip05 + self.q4_im05 )

        # Jacobian
        self.df_dphi_im10 = self.f.diff(self.phi_im10)
        self.df_dphi_ip10 = self.f.diff(self.phi_ip10)
        self.df_dphi_i = self.f.diff(self.phi_i)

        # geometry: nodes, x coordinate
        self.num_of_nodes = 1
        self.x_pos = [0.0]
        
        # geometry: elements, dx, material, epsilion, rho, doping
        self.num_of_elements = 0
        self.dx_pos = []
        self.mat = []
        self.ep = []
        self.rho = []
        self.dp = []

        # debugging
        if True:
            print(self.d_ip05)
            print(self.d_im05)
            print(self.q1_ip05)
            print(self.q1_im05)
            print(self.q2_ip05)
            print(self.q2_im05)
            print(self.q3_ip05)
            print(self.q3_im05)
            print(self.q4_ip05)
            print(self.q4_im05)
            print(self.f)
            print(self.df_dphi_im10)
            print(self.df_dphi_ip10)
            print(self.df_dphi_i)
            print('')


    def add_node(self, dx, mat, ep_k, rho, doping):
        # geometry: nodes, x coordinate
        self.num_of_nodes += 1
        self.x_pos.append(self.x_pos[-1] + dx)

        # geometry: elements, dx, mat, epsilion, rho, doping
        self.num_of_elements += 1
        self.dx_pos.append(dx)
        self.mat.append(mat)
        self.ep.append(ep_k * self.fc.ep0)
        self.rho.append(rho)
        self.dp.append(doping)

        # debugging
        if False:
            output_string  = '%i %i > ' % (self.num_of_nodes, self.num_of_elements)
            output_string +=  'x_pos = %s > ' % self.x_pos
            output_string +=  'dx_pos = %s > ' % self.dx_pos
            output_string +=  'mat = %s > ' % self.mat
            output_string +=  'ep = %s > ' % self.ep
            output_string +=  'rho = %s > ' % self.rho
            output_string +=  'doping = %s' % self.dp
            print(output_string)
            print('')


    def newton_raphson_method(self, temp, left_phi, right_phi):
        # phi
        self.phi = np.zeros(self.num_of_nodes, dtype=float)

        # f0
        self.f0 = np.zeros(self.num_of_nodes, dtype=float)
        
        # Jacobian matrix
        self.J = np.zeros([self.num_of_nodes, self.num_of_nodes], dtype=float)

        # delta phi
        self.dphi = np.zeros(self.num_of_nodes, dtype=float)

        # Dirichlet boundary conditions
        self.f0[0] = 0.0
        self.J[0, 0] = 1.0
        self.phi[0] = left_phi
        
        self.f0[-1] = 0.0
        self.J[-1, -1] = 1.0
        self.phi[-1] = right_phi

        #  visualization
        fig, ax = plt.subplots(1, 2)
        
        # loop
        nrm_legend = []
        
        for loop in range(10):
            #
            nrm_legend.append('%s' % loop)

            # Dirichlet boundary conditions
            self.f0[0] = 0.0
            self.J[0, 0] = 1.0
            self.phi[0] = left_phi
        
            self.f0[-1] = 0.0
            self.J[-1, -1] = 1.0
            self.phi[-1] = right_phi
            
            #
            for each_f in range(self.num_of_nodes):
                #
                if (each_f != 0) and (each_f != (self.num_of_nodes-1)):
                    #
                    vals = {}
                    vals[self.T] = temp + 273.15                    # Kelvin
                    # element > dx
                    vals[self.dx_im05] = self.dx_pos[each_f-1]      # dx
                    vals[self.dx_ip05] = self.dx_pos[each_f+0]      # dx
                    # element > electric permittivity
                    vals[self.ep_im05] = self.ep[each_f-1]          # electric permittivity
                    vals[self.ep_ip05] = self.ep[each_f+0]          # electric permittivity
                    # element > fixed charge
                    vals[self.rho_im05] = self.rho[each_f-1]        # fixed charge
                    vals[self.rho_ip05] = self.rho[each_f+0]        # fixed charge
                    # element > doping
                    vals[self.dp_im05] = self.dp[each_f-1]          # doping
                    vals[self.dp_ip05] = self.dp[each_f+0]          # doping
                    # nodes > phi_{i-1}, phi_{i}, phi_{i+1} 
                    vals[self.phi_im10] = self.phi[each_f-1]        # phi_{i-1}
                    vals[self.phi_i] = self.phi[each_f+0]           # phi_{i}
                    vals[self.phi_ip10] = self.phi[each_f+1]        # phi_{i+1}
                    #
                    self.f0[each_f] = self.f.evalf(subs=vals)
                    #
                    self.J[each_f, each_f-1] = self.df_dphi_im10.evalf(subs=vals)
                    self.J[each_f, each_f+0] = self.df_dphi_i.evalf(subs=vals)
                    self.J[each_f, each_f+1] = self.df_dphi_ip10.evalf(subs=vals)

            #
            self.dphi = np.linalg.solve(self.J, -self.f0)
            print(self.dphi)
            #
            self.phi = self.phi + self.dphi

            #
            print(loop, np.max(self.dphi))

            # visualization
            ax[0].plot(self.phi)
            ax[1].plot((self.phi[1:]-self.phi[:-1])/self.dx_pos)

        #
        ax[0].legend(nrm_legend)
        ax[1].legend(nrm_legend)
        plt.show()
        


    def governing_fdm(self):

        print('num of elements = %i' % self.num_of_elements)
        print('num of nodes = %i' % self.num_of_nodes)
        print('')

        for each_node in range(1, self.num_of_nodes-2):
            if each_node == 1:
                vals = {}
                vals[self.T] = 300.0
                vals[self.dx_im05] = self.dx_pos[each_node-1]
                vals[self.dx_ip05] = self.dx_pos[each_node]
                vals[self.ep_im05] = self.ep[each_node-1]
                vals[self.ep_ip05] = self.ep[each_node]
                vals[self.rho_im05] = self.rho[each_node-1]
                vals[self.rho_ip05] = self.rho[each_node]
                vals[self.dp_im05] = self.dp[each_node-1]
                vals[self.dp_ip05] = self.dp[each_node]
                vals[self.phi_im10] = 1.0
                vals[self.phi_i] = 0.0
                vals[self.phi_ip10] = 0.0
                print(each_node)
                print(vals)
                print(self.f.evalf(subs=vals))
                print(self.df_dphi_im10.evalf(subs=vals))
                print(self.df_dphi_ip10.evalf(subs=vals))
                print(self.df_dphi_i.evalf(subs=vals))
            




#
# MAIN
#

#
# Newton method
#

dopant_density_array = [1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
electric_potential_array = []
iteration_array = []

newton_method = NEWTON()

for index, dopant_density in enumerate(dopant_density_array):
    #
    error = 1.0
    iteration = 0

    #
    if index == 0:
        newton_method.set_expression(op_temp=300, dopant_density=dopant_density)
        phi = 0.0
        while error > 1e-3:
            phi_prev = phi
            phi = newton_method.newton_method(phi_prev)
            error = np.abs(phi - phi_prev)
            iteration += 1
        iteration_array.append(iteration)
        electric_potential_array.append(phi)

    else:
        newton_method.set_expression(op_temp=300, dopant_density=dopant_density)
        phi = electric_potential_array[-1]
        while error > 1e-3:
            phi_prev = phi
            phi = newton_method.newton_method(phi_prev)
            error = np.abs(phi - phi_prev)
            iteration += 1
        iteration_array.append(iteration)
        electric_potential_array.append(phi)
            
    #
    print('%i, %.6f' % (iteration, phi) )

#
# Newton-Raphson method
#

newton_raphson_method = NEWTON_RAPHSON()
newton_raphson_method.set_expression()

phi0 = [1, 2, 3]

for cnt in range(10):
    if cnt == 0:
        phi_new = newton_raphson_method.newton_raphson_method(phi0)
    else:
        phi_new = newton_raphson_method.newton_raphson_method(phi_new)

    print(cnt+1, phi_new)

#
# Nonlinear Poisson equation
#

npe = N_POISSON_EQ()

for cnt in range(20):
    npe.add_node(dx=1e-10, mat='O', ep_k=3.9, rho=0.0, doping=0.0)
for cnt in range(60):
    npe.add_node(dx=1e-10, mat='S', ep_k=11.7, rho=0.0, doping=1e24)
for cnt in range(20):
    npe.add_node(dx=1e-10, mat='O', ep_k=3.9, rho=0.0, doping=0.0)

#npe.governing_fdm()
npe.newton_raphson_method(temp=25.0, left_phi=0.0, right_phi=1.0)




