import numpy as np

def legendre_basis(deg=2):
    basis = [
        lambda x:np.ones_like(x),
        lambda x:x,
        lambda x:(3*x**2-1)/2,
        lambda x:(5*x**3-3*x)/2,
        lambda x:(35*x**4-30*x**2+3)/8,
        lambda x:(63*x**5-70*x**3+15*x)/8
    ]
    Dbasis = [
        lambda x:np.zeros_like(x),
        lambda x:np.ones_like(x),
        lambda x:3*x,
        lambda x:(5*3*x**2-3)/2,
        lambda x:(35*4*x**3-30*2*x)/8,
        lambda x:(63*5*x**4-70*2*x**2+15)/8
    ]
    return basis[:deg],Dbasis[:deg]

def baseline_basis(deg = 2):
    basis = [
        lambda x:0.5-0.5*x,
        lambda x:0.5+0.5*x,
    ]
    Dbasis = [
        lambda x:-0.5*np.ones_like(x),
        lambda x:0.5*np.ones_like(x),
    ]
    return basis[:deg],Dbasis[:deg]


def minmod_limiter(a,b,c):
    if np.sign(a) == np.sign(b) == np.sign(c):
        return np.sign(a)*np.min([np.abs(a),np.abs(b),np.abs(c)])
    else:
        return 0

def TVB_limiter(a,b,c,h,M):
    if np.abs(a) <= M*h**2:
        return a
    else:
        return minmod_limiter(a, b, c)