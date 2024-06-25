import numpy as np 
from computeDnTot import computeDnTot
from computeDiTot import computeDiTot
import logging

def computeDTot(S, Nci, Fci, Di, Rs, RTT, Ne, lambd, M, Rsd = np.empty(0)):

    # compute cloud-edge traffic
    max_delay = 1e6 # max delay used to avoid inf problem during optimization
    Dn_tot, rhonce = computeDnTot(S, Nci, Fci, Rs, RTT, Ne, lambd, M, Rsd)
    Di_tot = computeDiTot(Nci, Di)
    if rhonce == 1:
        logging.debug(f"computeDTot: inf network delay")
    return min(Dn_tot + Di_tot,max_delay), Di_tot, Dn_tot, rhonce