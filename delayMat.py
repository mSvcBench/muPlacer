import numpy as np
from buildFci import buildFci
from computeNcMat import computeNcMat
from computeDi import computeDi
from netdelay import netdelay

#   Compute the average delay Dm when the system gets state S

#   S : state vector
#   RTT : RTT edge-cloud
#   Ne : edge-cloud bit rate
#   lambd : average user request frequency
#   Rs : byte lenght of the response of microservice instance-set
#   Fcm : call frequency matrix
#   Fci : call frequency matrix edge+cloud
#   M : number of microservices
#   Di : matrix of delays introduced by each microservice instance
#   e: number of datacenters
#   Dn : matrix of network delays

def delayMat(S, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, e):
    MN = M * e   # edge+cloud microservice instance-sets
    Fci = np.matrix(buildFci(S, Fcm, M, e))
    Nc = computeNcMat(Fci, M, e)
    Di = computeDi(Nc, Rcpu, Rcpu_req, lambd, M, e) if np.sum(Rcpu_req) > 0 else np.zeros(2*M) 
    Dn, Tnce = netdelay(S, RTT, Ne, lambd, Rs, Fci, Nc, M, e) 
    
    H = -(Fci.T)
    N = Di
    for i in range(MN):
        H[i, i] = 1
        for j in range(MN):
            if i == j or Fci[i, j] == 0:
                continue
            N[i] = N[i] + Fci[i, j] * Dn[i, j]
            
    H_inv = np.linalg.inv(H)
    Dm = np.array(np.dot(N, H_inv))
    Dm = Dm.flatten()
    d = Dm[MN-1]
    if np.isnan(d):
        d = float('inf')

    return d