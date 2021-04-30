import numpy as np

def legendre_basis(deg=2):
    basis = [
        lambda x:1,
        lambda x:x,
        lambda x:(3*x**2-1)/2,
        lambda x:(5*x**3-3*x)/2,
        lambda x:(35*x**4-30*x**2+3)/8,
        lambda x:(63*x**5-70*x**3+15)/8
    ]
    return basis[:deg]