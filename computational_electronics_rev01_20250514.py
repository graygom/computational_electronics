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

