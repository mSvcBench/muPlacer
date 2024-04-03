import numpy as np

#   Compute average number of time a microservice is called in a request (Nc)

#   M : number of microservices
#   e : number of datacenters
#   Fci : service mesh call frequency matrix 


def computeNcMat(Pc, M, e):
    MN = len(Pc)
    H = -Pc
    np.fill_diagonal(H, 1)
    if e > 1:
        Ubit = np.arange(2, e+1) * M  # user position in the state vector
    else:
        Ubit = MN
    N = np.zeros(MN)
    N[Ubit-1] = 1
    H_inv = np.linalg.inv(H)
    Nc = np.array(np.dot(N, H_inv))
    Nc = Nc.flatten()
    return Nc