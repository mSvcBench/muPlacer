import numpy as np
from mI2mV import mI2mV

#   Comupute the network delay matrix Dn

#   S : state vector
#   RTT : RTT edge-cloud
#   B : edge-cloud bit rate
#   lambd : average user request frequency
#   L : byte lenght of the response of microservice instance-set
#   Fi : service mesh call frequency matrix 
#   N : number of time a microservice instance-set is called per request
#   M : number of microservice instance-sets
#   e : number of datacenters

def netdelay(S, RTT, B, lambd, L, Fi, N, M, e):
    MN = M * e  # edge+cloud microservice instance-sets
    Tnce = np.zeros(e - 1)  # Inizialization of array for volume of cloud-edge traffic
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e + 1):  
                if di == h and dj == 1 and S[I]==1:
                    Tnce[h - 2] = Tnce[h - 2] + lambd * N[I] * Fi[I, J] * L[J] * 8 # Compute Tnce
    
    rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e+1):
                if di == h and dj == 1 and Fi[I, J]>0:
                    Dn[I,J] = RTT + ((L[J] * 8 / B) + 0.015) / (1 - rhonce[h - 2]) 
                    continue
    return Dn, Tnce

def netdelay2(S, RTT, B, lambd, L, Fi, N, M):
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    for i in S_edge_id:
        for j in S_not_edge_id:
            Tnce = Tnce + lambd * N[M+i] * Fi[M+i, j] * L[j] * 8 # Compute Tnce
    
    rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for i in S_edge_id:
        for j in S_not_edge_id:
            if Fi[M+i,j]>0:
                Dn[M+i,j] = RTT + ((L[j] * 8 / B) + 0.015) / (1 - rhonce) 
    return Dn, Tnce

    