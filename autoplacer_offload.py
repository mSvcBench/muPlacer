import datetime
import numpy as np
from heuristic_autoplacer_offload import heuristic_autoplacer_offload

def autoplacer_offload(Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT):
    x = datetime.datetime.now().strftime('%d-%m_%H:%M:%S')
    filename = f'offload_{x}.mat'
    #np.save(filename, arr=[Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT])

    app_edge = np.append(app_edge, 1)  # add the user in app_edge (user is in the edge cluster)
    e = 2  # number of data center
    Ne = 1e9  # cloud-edge bit rate
    Ubit = np.arange(1, e+1) * M  # user position in the state vector
    Ce = np.inf
    Me = np.inf

    Rs = np.append(Rs, 0)  # add the user in Rs vector
    Rs = np.tile(Rs, e)  # edge and cloud microservices response size, len(Rs)=20

    Rcpu_req = np.tile(np.zeros(int(M)), e)  # seconds of CPU per request (set to zero for all microservices)
    Rcpu_req[int(Ubit[0])-1] = 0  # not used in our case, only if Rcpu_req is not filled with all zeros
    Rcpu_req[int(Ubit[1])-1] = 0  # not used in our case, only if Rcpu_req is not filled with all zeros

    best_S, best_dw, Dn, Tnce, delta = heuristic_autoplacer_offload(Fcm_nocache, RTT, Rcpu_req, Rcpu, Rmem, Ce, Me, Ne, lambd, Rs, M, 0, 1, 2, app_edge, min_delay_delta)
    best_S_edge = best_S[M:2*M]  # takes only edge microservices
    return best_S_edge, delta