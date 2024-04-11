import datetime
import numpy as np
from heuristic_unoffload import heuristic_unoffload

def unoffload(Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, max_delay_delta, RTT, Ne):
    #x = datetime.datetime.now().strftime('%d-%m_%H:%M:%S')
    #filename = f'unoffload_{x}.mat'
    #np.save(filename, arr=[Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT])

    app_edge = np.append(app_edge, 1)  # Add the user in app_edge (user is in the edge cluster)
    e = 2  # Number of datacenters
    Ubit = np.arange(1, e+1) * M  # User position in the state vector
    Ce = np.inf # CPU capacity of edge datacenter
    Me = np.inf # Memory capacity of edge datacenter
    Rs = np.append(Rs, 0)  # Add the user in Rs vector
    Rs = np.tile(Rs, e)  # Expand the Rs vector to fit the number of data centers

    Rcpu_req = np.tile(np.zeros(int(M)), e)  # Seconds of CPU per request (set to zero for all microservices)
    # Seconds of CPU per request for the user
    Rcpu_req[int(Ubit[0])-1] = 0 
    Rcpu_req[int(Ubit[1])-1] = 0

    best_S = heuristic_unoffload(Fcm_nocache, RTT, Rcpu_req, Rcpu, Rmem, Ce, Me, Ne, lambd, Rs, M, 0, 1, 2, app_edge, max_delay_delta)
    best_S_edge = best_S[M:2*M]  # Takes only edge part of the new state vector
    return best_S_edge