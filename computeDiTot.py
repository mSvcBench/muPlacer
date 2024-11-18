import numpy as np 

def computeDiTot(N, Di):

    # compute total internal delay introduced by microserive per user request
    # N : number of instance call per user request of the current state
    # Di : internal delay of microservices
    
    return (np.sum(np.multiply(N,Di)))
