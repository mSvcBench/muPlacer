import numpy as np 

def computeDnTot(S, Nci, Fci, Rs, RTT, Ne, lambd, M):

    # compute cloud-edge traffic
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    for i in S_edge_id:
        for j in S_not_edge_id:
            Tnce = Tnce + lambd * Nci[M+i] * Fci[M+i, j] * Rs[j] * 8 # Compute Tnce
    
    rhonce = min(Tnce / Ne, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for i in S_edge_id:
        for j in S_not_edge_id:
            if Fci[M+i,j]>0:
                Dn[M+i,j] = RTT + ((Rs[j] * 8 / Ne) + 0.015) / (1 - rhonce) 
    
    
    Dn_tot = np.sum(np.multiply((Nci[M:].reshape(M,1)),(np.sum(np.multiply(Fci[M:,:M],Dn[M:,:M]),axis=1))))

    return Dn_tot