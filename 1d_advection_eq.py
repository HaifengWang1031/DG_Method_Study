import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""
u_t + f(u)_x  = 0, 0<=x<=1, t_final = 2*pi,
u(x,0) = u_0 = sin(2*pi*x),
with periodic boundary conditions.
In this case, we assume the flux function f(u) = cu, where c is constant.
"""
c = 1 # velocity

tau = 2*np.pi

N = 2 # element polynomial degree 
basis = [lambda x:1-x,lambda x:x] # basis function

K = 32 # element number
xNode,step =np.linspace(0,1,K+1,retstep=True)

xh = np.vstack([xNode[0:-1],xNode[1:]]).T

def multi_wave(x):
    if 0.1 < x <=0.2:
        return 10*(x-0.1)
    elif 0.2 < x <= 0.3:
        return 10*(0.3-x)
    elif 0.4< x <=0.6:
        return 1
    elif 0.8< x <=0.9:
        return 100*(x-0.8)*(0.9-x)
    else:
        return 0

# u0 = lambda x:np.sin(tau*x)
u0 = multi_wave


# 1. calculate initial basis weight 
ExactRHS = np.zeros((N,K))
for n in range(N):
    for k in range(K):
        ExactRHS[n][k] = integrate.quad(lambda x:u0(x)*basis[n]((x - xh[k][0])/step),xh[k][0],xh[k][1])[0]

MassMatrix = np.zeros((N,N))
for n1 in range(N):
    for n2 in range(N):
        MassMatrix[n1][n2] = integrate.quad(lambda x:basis[n2](x)*basis[n1](x),0,1)[0]

BasisWeights = np.linalg.solve(MassMatrix,ExactRHS/step)

BasisWeights = np.reshape(BasisWeights,(N*K,1),order="F")

# 2.the semi_discrete system can be writen as u_t = L(u). Let's assemble the operator L.
up_wind_flux = c*np.array([[1,0,0],[0,0,-1]])
RHSIntegrals = np.array([[0,-1/2,-1/2],[0,1/2,1/2]])
Stencil = np.linalg.solve(MassMatrix,(up_wind_flux+RHSIntegrals)/step)

SemiMatrix = np.zeros([N*K,N*K])
for i in range(0,N*K-N,N):
    SemiMatrix[i:i+N,i:i+N+1] = Stencil
SemiMatrix = np.roll(SemiMatrix,-1,axis=1) #apply periodic BC
SemiMatrix[-N:,-N-1:] = Stencil

print(RHSIntegrals)
print(Stencil)
print(MassMatrix)
print(SemiMatrix)

# 3. Time Discretization
delta_T = 0.001
final_T = 3
saved_u = BasisWeights 
for t in np.arange(0,final_T,delta_T):
    # TVD-RK3
    w1 = BasisWeights + np.matmul(SemiMatrix,BasisWeights)*delta_T
    w2 = 3/4*BasisWeights + 1/4*(w1 + np.matmul(SemiMatrix,w1)*delta_T)
    BasisWeights = 1/3*BasisWeights + 2/3*(w2 +np.matmul(SemiMatrix,w2)*delta_T)
    saved_u =  np.concatenate((saved_u,BasisWeights),axis=1)

saved_u = saved_u.T
saved_u = np.reshape(saved_u,(len(saved_u),-1,2))

# 交互模式显示动图
# plt.ion()
# for t in range(len(saved_u)):
#     for i in range(K):
#         plt.plot(xh[i],saved_u[t,i])
#     plt.pause(0.001)
#     plt.clf()
# plt.ioff()

fig  = plt.figure()
ims = []
for t in range(0,len(saved_u),10):
    im = plt.plot(xh.T,saved_u[t].T)
    ims.append(im)

# ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000)
# ani.save("test.gif",writer='pillow')
# plt.show()