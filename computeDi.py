import numpy as np 

#    Compute the internal microservices function delay matrix Di

#   M/G/1 processor sharing
#   Rcpu : cpu requested by microservice i
#   lambd : user request frequency
#   Rcpu_req : CPU seconds consumed to execute internal function
#   Nc : number of time a microservice instance is called per request
#   M : number of microservices
#   e : number of datacenters

def computeDi(Nc, Rcpu, Rcpu_req, lambd, M, e):

    Ubits = [i * M for i in range(1, e+1)]
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_c = np.minimum(lambd * Nc * Rcpu_req / Rcpu, 1)
        rho_c[np.isnan(rho_c)] = 1
        Di = np.array((Rcpu_req / Rcpu) / (1 - rho_c))
    for ubit in Ubits:
        Di[ubit-1] = 0
    return Di