import numpy as np

#   Compute average number of time a microservice/instance is called per request

#   M : number of microservices
#   e : number of datacenters. e=1 compute Nc per microservice, e=2 compute Nc per instance
#   Fc : call frequency matrix 


def computeNc(Fc, M, e):
    MN = len(Fc)
    H = -Fc.copy()
    np.fill_diagonal(H, 1)
    if e > 1:
        Ubit = np.arange(2, e+1) * M  # user position in the state vector
    else:
        Ubit = MN
    N = np.zeros(MN)
    N[Ubit-1] = 1
    H_inv = np.linalg.inv(H)
    Nc = np.dot(N, H_inv)
    Nc = np.array(Nc).flatten()
    return Nc