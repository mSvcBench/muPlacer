import numpy as np
from mI2mV import mI2mV

#   Comupute the network delay matrix Dn

#   S : state vector
#   RTT : RTT edge-cloud
#   Ne : edge-cloud bit rate
#   lambd : average user request frequency
#   Rs : byte lenght of the response of microservice instance-set
#   Fci : service mesh call frequency matrix 
#   Nc : number of time a microservice instance-set is called per request
#   M : number of microservice instance-sets
#   e : number of datacenters

def netdelay(S, RTT, Ne, lambd, Rs, Pci, Nc, M, e):
    MN = M * e  # edge+cloud microservice instance-sets
    Tnce = np.zeros(e - 1)  # Inizialization of array for volume of cloud-edge traffic
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e + 1):  
                if di == h and dj == 1 and S[I]==1:
                    Tnce[h - 2] = Tnce[h - 2] + lambd * Nc[I] * Pci[I, J] * Rs[J] * 8 # Compute Tnce
    
    rhonce = min(Tnce / Ne, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e+1):
                if di == h and dj == 1 and Pci[I, J]>0:
                    Dn[I,J] = RTT + ((Rs[J] * 8 / Ne) + 0.015) / (1 - rhonce[h - 2]) 
                    continue
    return Dn, Tnce

def netdelay2(S, RTT, Ne, lambd, Rs, Fci, Nc, M):
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    for i in S_edge_id:
        for j in S_not_edge_id:
            Tnce = Tnce + lambd * Nc[M+i] * Fci[M+i, j] * Rs[j] * 8 # Compute Tnce
    
    rhonce = min(Tnce / Ne, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for i in S_edge_id:
        for j in S_not_edge_id:
            if Fci[M+i,j]>0:
                Dn[M+i,j] = RTT + ((Rs[j] * 8 / Ne) + 0.015) / (1 - rhonce) 
    return Dn, Tnce

    