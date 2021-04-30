import numpy as np
import scipy.integrate as integrate
from utils import *


class CL_Solver:
    def __init__(self,flux_type,basis_order=2,space_interval=[0,1],ele_num = 32,final_T=2):
        self.flux_type = flux_type
        
        self.N = basis_order
        self.basis = legendre_basis(self.N) 
        
        self.space_interval = space_interval 
        self.K = ele_num
        self.x_node = np.linspace(*self.space_interval,num=self.K+1)
        self.delta_x = np.diff(self.x_node)
        self.x_h = np.vstack([self.x_node[0:-1],self.x_node[1:]]).T

        self.final_T = final_T

    def mass_matrix(self):
        pass

    def RHS_operator(self):
        pass

    def trans_init_weight(self,init_func):
        pass

solver = CL_Solver("a")
print(solver.delta_x)