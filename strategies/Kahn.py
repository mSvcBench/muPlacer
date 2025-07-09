import numpy as np
import logging
import sys

from numpy import inf
from collections import deque
from utils import buildFi, computeDTot, computeN, computeCost, computeResourceShift, numpy_array_to_list


# Set up logger
logger = logging.getLogger('Kahn_logger')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s SBMP offload %(levelname)s %(message)s'))
logger.propagate = False

def Kahn_heuristic(params):
    S_edge_old = params['S_edge_b']
    Ucpu_old = params['Ucpu']
    Umem_old = params['Umem']
    Fm = params['Fm']
    M = params['M']
    lambd = params['lambd']
    L = params['L']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target']
    delay_increase_target = params['delay_increase_target']
    RTT = params['RTT']
    B = params['B']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']
    Cost_cpu_cloud = params['Cost_cpu_cloud']
    Cost_mem_cloud = params['Cost_mem_cloud']
    Cost_network = params['Cost_network']

    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)

    global_HPA_cpu_th = params['global_HPA_cpu_th'] if 'global_HPA_cpu_th' in params else None

    L = np.tile(L, 2)  # Expand the L vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    
    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fi_old = np.matrix(buildFi(S_b_old, Fm, M))
    N_old = computeN(Fi_old, M, 2)
    delay_old,_,_,rhoce_old = computeDTot(S_b_old, N_old, Fi_old, Di, L, RTT, B, lambd, M)
    Cost_old = computeCost(Ucpu_old, Umem_old, Qcpu, Qmem , Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud,rhoce_old*B,Cost_network, global_HPA_cpu_th)[0] # Total cost of old state
    N = computeN(Fm, M, 1)
    delay_decrease_new = 0
    S_b_new = S_b_old.copy()


    # Create Kahn's sorting
    n = Fm.shape[0]
    in_degree = np.count_nonzero(Fm, axis=0)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    kahn_topo_order = []
    kahan_value = np.zeros(M)
    k=0
    while queue:
        u = queue.popleft()
        kahn_topo_order.append(u)
        kahan_value[u] = k
        k += 1
        for v in range(n):
            if Fm[u][v] > 0:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if delay_decrease_target > 0:
        ## OFFLOAD ##
        while delay_decrease_target > delay_decrease_new:
            ## find the first microservice of topo_order not present in the edge ##
            ms_candidates = np.argwhere(S_b_new[M:]==0).flatten() # ms not at the edge are candidates for offloading
            best_ms_idx = np.argmin(kahan_value[ms_candidates])  # find the microservice with the lowest Kahn's value
            best_ms = ms_candidates[best_ms_idx]  # get the microservice id
            S_b_new[best_ms+M] = 1
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0] 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                # all instances at the edge
                break

            # S_b_edge_kahn_sorted = S_b_new[M + np.array(kahn_topo_order)]  # S edge placement vector sorted by Kahn's topo order
            # idxs = np.where(S_b_edge_kahn_sorted == 0)[0] # first element of S not at the edge
            # first_idx = idxs[0] if idxs.size > 0 else None
            # candidate_ms = kahn_topo_order[first_idx]           
            # S_b_new[candidate_ms+M] = 1
            # Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            # N_new = computeN(Fi_new, M, 2)
            # delay_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0] 
            # delay_decrease_new = delay_old - delay_new
            # if np.all(S_b_new[M:] == 1):
            #     # all instances at the edge
            #     break
    
    ## UNOFFLOAD  ##
    else:
        
        while True:
            ms_edge = np.argwhere(S_b_new[M:]==1).flatten()
            ms_candidates = np.argwhere(S_b_new[M:2*M-1]==1).flatten() 
            best_ms_idx = np.argmax(kahan_value[ms_candidates])  # find the microservice with the lowest Kahn's value
            best_ms = ms_candidates[best_ms_idx]  # get the microservice id
            S_b_new[best_ms+M] = 0
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0] 
            if delay_new - delay_old > delay_increase_target:
                # if the delay increase is above the target, stop unoffloading
                S_b_new[best_ms+M] = 1 # cancell the unoffload
                break
            if np.all(S_b_new[M:2*M-1] == 0):
                # no edge microservice not possible to unoffload more
                break

    # compute final values
    Ucpu_new = np.zeros(2*M)
    Umem_new = np.zeros(2*M)
    Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
    N_new = computeN(Fi_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)            
    delay_decrease_new = delay_old - delay_new
    np.copyto(Ucpu_new,Ucpu_old) 
    np.copyto(Umem_new,Umem_old)
    computeResourceShift(Ucpu_new, Umem_new, N_new, Ucpu_old, Umem_old, N_old) 
    Cost_new, Cost_new_edge, Cost_new_cloud, Cost_traffic_new = computeCost(Ucpu_new, Umem_new, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_new * B, Cost_network, global_HPA_cpu_th) # Total cost of new state
    cost_increase_new = Cost_new - Cost_old 

    result_metrics = dict()
    
    # extra information
    result_metrics['S_edge_b'] = S_b_new[M:].astype(int)
    result_metrics['Cost'] = Cost_new
    result_metrics['Cost_edge'] = Cost_new_edge
    result_metrics['Cost_cloud'] = Cost_new_cloud
    result_metrics['Cost_traffic'] = Cost_traffic_new
    result_metrics['delay_decrease'] = delay_decrease_new
    result_metrics['delay_increase'] = -delay_decrease_new
    result_metrics['cost_increase'] = cost_increase_new
    result_metrics['cost_decrease'] = -cost_increase_new
    result_metrics['Ucpu'] = Ucpu_new
    result_metrics['Umem'] = Umem_new
    result_metrics['Fi'] = Fi_new
    result_metrics['N'] = N_new
    result_metrics['delay'] = delay_new
    result_metrics['di'] = di_new
    result_metrics['dn'] = dn_new
    result_metrics['rhoce'] = rhoce_new
    
    # required return information
     
    result_cloud = dict()
    result_cloud['to-apply'] = list()
    result_cloud['to-delete'] = list()
    result_cloud['placement'] = numpy_array_to_list(np.argwhere(S_b_new[:M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"

    result_edge = dict()
    result_edge['to-apply'] = numpy_array_to_list(np.argwhere(S_b_new[M:]-S_b_old[M:]>0))
    result_edge['to-delete'] = numpy_array_to_list(np.argwhere(S_b_old[M:]-S_b_new[M:]>0))
    result_edge['placement'] = numpy_array_to_list(np.argwhere(S_b_new[M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_metrics['delay_decrease'] < delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    result_return.append(result_metrics)
    return result_return

    return result



