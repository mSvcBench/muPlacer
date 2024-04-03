import numpy as np
from buildFci import buildFci
from computeNcMat import computeNcMat
from computeDi import computeDi
from netdelay import netdelay

#   Compute the average delay Dm when the system gets state S

#   S : state vector
#   RTT : RTT cloud edge
#   Ne : edge-cloud bit rate
#   lambd : user request frequency
#   Rs : byte length of the response of microservices
#   Fcm : call frequency matrix
#   Fci : service mesh call frequency matrix 
#   M : number of microservices
#   Di : internal function execution time matrix
#   e: number of datacenters

def delayMat(S, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, e):
    MN = M * e   # mesh nodes
    Fci = np.matrix(buildFci(S, Fcm, M, e))
    Nc = computeNcMat(Fci, M, e)
    Di = computeDi(Nc, Rcpu, Rcpu_req, lambd, M, e)
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

    return d, Dn, Tnce, Di, Nc


    # MN = M * e   # mesh nodes
    # #Ns = 2 ** MN   # number of possible states of the state vector S
    # Pci = np.matrix(buildPci(S, Fcm, M, e, False))
    # Nc = computeNcMat(Pci, M, e)
    # Di = computeDi(Nc, Rcpu, Rcpu_req, lambd, M, e)
    # Dn, Tnce, Tnec = netdelay(S, RTT, Ne, lambd, Rs, Pci, Nc, M, e)   # i,j network delay 
    
    # H = -(Pci.T)
    # N = Di
    # for i in range(MN):
    #     H[i, i] = 1
    #     for j in range(MN):
    #         if i == j or Pci[i, j] == 0:
    #             continue
    #         N[i] = N[i] + Pci[i, j] * Dn[i, j]
    # #Dm = N/H
    # H_inv = np.linalg.inv(H)
    # Dm = np.array(np.dot(N, H_inv))
    # Dm = Dm.flatten()
    # d = Dm[MN-1]
    # if np.isnan(d):
    #     d = float('inf')

    # return d, Dn, Tnce, Tnec, Di, Nc