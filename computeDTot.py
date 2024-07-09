import numpy as np 
from computeDnTot import computeDnTot
from computeDiTot import computeDiTot
import logging

def computeDTot(S, Nci, Fci, Di, Rs, RTT, Ne, lambd, M, Rsd = np.empty(0)):

    # compute average service delay measured at the ingress proxy of the edge data center
    
    # S : binary presence vector
    # Nci : average number of calls per user request per microservice
    # Fci : call frequency matrix
    # Di : internal delay introduced by microservices
    # Rs : response size of microservices
    # RTT : round trip time
    # Ne : network bandwidth
    # lambd : average number of requests per second
    # M : number of microservices
    # Rsd : duration of cloud edge data transfer for Rs
    
    max_delay = 1e5 # max delay used to avoid inf problem during optimization
    Dn_tot, rhonce = computeDnTot(S, Nci, Fci, Rs, RTT, Ne, lambd, M, Rsd)
    Di_tot = computeDiTot(Nci, Di)
    if rhonce == 1:
        logging.debug(f"computeDTot: inf network delay")
    return min(Dn_tot + Di_tot,max_delay), Di_tot, Dn_tot, rhonce