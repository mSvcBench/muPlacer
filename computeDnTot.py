import numpy as np 

def computeDnTot(S, N, Fi, L, RTT, B, lambd, M, Ld = np.empty(0)):

    # compute cloud-edge traffic
    
    # S : binary presence vector
    # N : average number of calls per user request per microservice
    # Fi : call frequency matrix
    # L : response length of microservices
    # RTT : round trip time
    # B : network bandwidth
    # lambd : average number of requests per second
    # M : number of microservices
    # Ld : duration of cloud edge data transfer for L 

    max_delay = 1e5 # max delay used to avoid inf problem during optimization
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    if Ld.size==0:
        Tnce = np.sum(np.multiply(np.multiply(Fi[M:,:M],np.repeat(N[M:].reshape(M,1),M,axis=1)),np.repeat(L[:M].reshape(1,M),M,axis=0)))*lambd*8
        rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection
        load_spread = 0
        rhonce_max = 1
        if rhonce < rhonce_max:
            load_spread = 1/(1 - rhonce)  # Load spread factor
        else:
            load_spread = 1e6*Tnce/B    # Load spread factor fostering solution with lower traffic
            # load_spread = (rhonce * 1/((1-rhonce_max)**2)) + ((1-2*rhonce_max)/((1-rhonce_max)**2))  # Load spread factor
        Dn = np.repeat(np.minimum(((L * 8 / B)*(load_spread)),max_delay).reshape(1,2*M),2*M,axis=0)+RTT
    else:
        Dn[M:,:M]=np.repeat(Ld.reshape(1,M),M,axis=0)
    Dn = np.repeat(np.minimum(((L * 8 / B)*(load_spread)),max_delay).reshape(1,2*M),2*M,axis=0)+RTT
    Dn_tot = np.sum(np.multiply((N[M:].reshape(M,1)),(np.sum(np.multiply(Fi[M:,:M],Dn[M:,:M]),axis=1))))

    return Dn_tot, rhonce 

# def computeDnTot_old(S, Nci, Fci, Rs, RTT, Ne, lambd, M, Rsd = np.empty(0)):

#     # compute cloud-edge traffic
    
#     # S : binary presence vector
#     # Nci : average number of calls per user request per microservice
#     # Fci : call frequency matrix
#     # Rs : response size of microservices
#     # RTT : round trip time
#     # Ne : network bandwidth
#     # lambd : average number of requests per second
#     # M : number of microservices
#     # Rsd : duration of cloud edge data transfer for Rs

#     max_delay = 1e5 # max delay used to avoid inf problem during optimization
#     MN = 2*M  # edge+cloud microservice instance-sets
#     Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
#     S_edge_id = np.argwhere(S[M:]==1).flatten()
#     S_not_edge_id = np.argwhere(S[M:]==0).flatten()
#     for i in S_edge_id:
#         for j in S_not_edge_id:
#             Tnce = Tnce + lambd * Nci[M+i] * Fci[M+i, j] * Rs[j] * 8 # Compute Tnce
    
#     rhonce = min(Tnce / Ne, 1)  # Utilization factor of the cloud-edge connection
#     #load_spread = 1/(1 - rhonce)  # Load spread factor
#     load_spread = 0
#     rhonce_max = 0.9
#     if rhonce < rhonce_max:
#         load_spread = 1/(1 - rhonce)  # Load spread factor
#     else:
#         load_spread = (rhonce * 1/((1-rhonce_max)**2)) + ((1-2*rhonce_max)/((1-rhonce_max)**2))  # Load spread factor
    
#     # Compute Dn
#     Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
#     for i in S_edge_id:
#         for j in S_not_edge_id:
#             if Fci[M+i,j]>0:
#                 if Rsd.size==0 or Rsd[j]==0:
#                     Dn[M+i,j] = RTT + min((Rs[j] * 8 / Ne)*(load_spread),max_delay) # M/M/1 with processor sharing. 
#                 else:
#                     Dn[M+i,j] = Rsd[j]  # Use the provided delay
    
#     Dn_tot = np.sum(np.multiply((Nci[M:].reshape(M,1)),(np.sum(np.multiply(Fci[M:,:M],Dn[M:,:M]),axis=1))))
#     Dn_tot_new, rhonce_new = computeDnTot_new(S, Nci, Fci, Rs, RTT, Ne, lambd, M, Rsd)

#     return Dn_tot, rhonce