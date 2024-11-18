import numpy as np 
from computeDnTot import computeDnTot
from computeDiTot import computeDiTot
import logging

def computeDTot(S, N, Fi, Di, L, RTT, B, lambd, M, Ld = np.empty(0)):

    # compute average service delay measured at the ingress proxy of the edge data center
    
    # S : binary presence vector
    # N : average number of calls per user request per microservice
    # Fi : call frequency matrix
    # Di : internal delay introduced by microservices
    # L : response length of microservices
    # RTT : round trip time
    # B : network bandwidth
    # lambd : average number of requests per second
    # M : number of microservices
    # Ld : duration of cloud edge data transfer for L
    
    max_delay = 1e5 # max delay used to avoid inf problem during optimization
    Dn_tot, rhonce = computeDnTot(S, N, Fi, L, RTT, B, lambd, M, Ld)
    Di_tot = computeDiTot(N, Di)
    if rhonce == 1:
        logging.debug(f"computeDTot: inf network delay")
    return min(Dn_tot + Di_tot,max_delay), Di_tot, Dn_tot, rhonce