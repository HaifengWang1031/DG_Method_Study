import numpy as np
import scipy.integrate as integrate
from utils import *
import matplotlib.pyplot as plt


class CL_Solver:
    """
    u_t + f(u)_x = 0, x \in space_interval, t \in [0,final_T]
    u(x,0) = u_0,
    with periodic boundary condition,

    The flux term can choose from
    1) scale linear advcection f(u) = cu
    2) Burgers f(u) = u^2/2
    3) Buckley-Leverett f(u) = u^2/(u^2+0.5*(1-u)^2)
    """
    def __init__(self,flux_type = 1,basis_order=2,space_interval=[0,1],ele_num = 32):
        self.flux_type = flux_type

        self.N = basis_order
        self.basis,self.Dbasis = legendre_basis(self.N)

        self.space_interval = space_interval
        self.K = ele_num
        self.x_node = np.linspace(*self.space_interval,num=self.K+1)
        self.delta_x = np.diff(self.x_node)
        self.x_h = np.vstack([self.x_node[0:-1],self.x_node[1:]]).T

        self.Mass_Matrix()
        self.RHS_operator()


    def Mass_Matrix(self):
        """
        mass_matrix = delta_x/2 * int_{-1,1} phi_i*phi_j dx
        """
        self.mass_matrix = np.empty((self.N,self.N))
        for n1 in range(self.N):
            for n2 in range(self.N):
                self.mass_matrix[n1,n2] = integrate.quad(lambda x:self.basis[n1](x)*self.basis[n2](x),-1,1)[0]

    def RHS_operator(self):
        """
        RHS_operator = mass_matrix\(RHS_integar + number_flux)
        """
        #here we use Lax-Friedrichs flux:f(u-,u+) = 1/2(f(u-)-f(u+)-\alpha(u+ - u-))

        #step 1: compute the local stencil
        if self.flux_type == 1:
            c = 1 # set velocity number
            f_l =np.array([basis(-1) for basis in self.basis]).reshape((1,-1))
            f_r = np.array([basis(1) for basis in self.basis]).reshape((1,-1))
            phi_l = np.array([basis(-1) for basis in self.basis]).reshape((-1,1))
            phi_r = np.array([basis(1) for basis in self.basis]).reshape(-1,1)

            num_flux =c*np.concatenate((f_r*phi_l,-f_r*phi_r,np.zeros((self.N,self.N))),axis = 1) \
                if c>=0 \
                else c*np.concatenate((np.zeros((self.N,self.N)),f_l*phi_l,-f_l*phi_r),axis = 1)

            RHS_integar = np.empty((self.N,self.N))
            for n1 in range(self.N):
                for n2 in range(self.N):
                    RHS_integar[n1][n2] = c*integrate.quad(lambda x:self.Dbasis[n1](x)*self.basis[n2](x),-1,1)[0]
            RHS_integar = np.concatenate((np.zeros((self.N,self.N)),RHS_integar,np.zeros((self.N,self.N))),axis = 1)

            Stencil = np.linalg.solve(self.mass_matrix,(RHS_integar+num_flux))

        #step 2: Assemble semi-discrete system and apply periodic BC

            SemiMatrix = np.zeros([self.N*self.K,self.N*(self.K+2)])

            for e,i in enumerate(range(0,self.N*self.K,self.N)):
                SemiMatrix[i:i+self.N,i:i+3*self.N] = Stencil/(self.delta_x[e]/2)

            # Apply periodic BC
            SemiMatrix[:,-2*self.N:-self.N]+= SemiMatrix[:,:self.N]
            SemiMatrix[:,self.N:2*self.N] += SemiMatrix[:,-self.N:]
            SemiMatrix = SemiMatrix[:,self.N:-self.N]

        elif self.flux_type == 2:
            pass
        elif self.flux_type == 3:
            pass
        else:
            raise RunTimeError("To be continue!")
        self.SemiMatrix = SemiMatrix


    def Limiter(self,BasisWeights):
        local_weight = np.reshape(BasisWeights,(self.K,self.N))
        #TODO
        #step 1. detection of trouble-cells

        #step 2. use a suitable limiter to reconstructing the polynomial solution

        ReconstructedWeight = local_weight.reshape(-1,1)
        return ReconstructedWeight

    def reset(self,init_func):
        ExactRHS = np.zeros((self.N,self.K))
        for n in range(self.N):
            for k in range(self.K):
                ExactRHS[n][k] = integrate.quad(lambda x:init_func(x)*self.basis[n]((2*x - (self.x_h[k][0]+self.x_h[k][1]))/self.delta_x[k]),self.x_h[k][0],self.x_h[k][1])[0]/(self.delta_x[k]/2)


        BasisWeights = np.linalg.solve(self.mass_matrix,ExactRHS)
        BasisWeights = np.reshape(BasisWeights,(self.N*self.K,1),order="F")
        self.BasisWeights = BasisWeights
        self.WeightContainer = self.BasisWeights

    def step(self,delta_t):
        # TDV-RK3 method
        w1 = self.BasisWeights + np.matmul(self.SemiMatrix,self.BasisWeights)*delta_t
        w2 = 3/4*self.BasisWeights + 1/4*(w1 + np.matmul(self.SemiMatrix,w1)*delta_t)
        self.BasisWeights = 1/3*self.BasisWeights + 2/3*(w2 +np.matmul(self.SemiMatrix,w2)*delta_t)
        self.BasisWeights = self.Limiter(self.BasisWeights)
        self.WeightContainer =  np.concatenate((self.WeightContainer,self.BasisWeights),axis=1)

    def draw_step(self,weights):
        for e,i in enumerate(range(0,len(weights),self.N)):
            x_e = np.linspace(self.x_h[e][0],self.x_h[e][1])
            result_local = np.zeros_like(x_e)
            for j in range(self.N):
                result_local += weights[i+j]*self.basis[j]((2*x_e - (self.x_h[e][0]+self.x_h[e][1]))/self.delta_x[e])
            plt.plot(x_e,result_local)

# initial wave for Linear advection equation
def sine_wave(x):
    return np.sin(10*np.pi*x)

def multi_wave(x):
    if 0.2 < x <=0.3:
        return 10*(x-0.2)
    elif 0.3 < x <= 0.4:
        return 10*(0.4-x)
    elif 0.6< x <=0.8:
        return 1
    elif 1< x <=1.2:
        return 100*(x-1)*(1.2-x)
    else:
        return 0

if __name__ == "__main__":

    solver = CL_Solver(flux_type = 1,basis_order=2,space_interval=[0,1.4],ele_num = 100)

    plt.ion()
    solver.reset(multi_wave)
    for _ in range(1400):
        solver.draw_step(solver.BasisWeights)
        solver.step(0.001)
        plt.pause(0.001)
        plt.clf()
    plt.ioff()
