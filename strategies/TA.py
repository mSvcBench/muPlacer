import numpy as np
import logging
import sys

from numpy import inf
from collections import deque
from utils import buildFi, computeDTot, computeN, computeCost, computeResourceShift, numpy_array_to_list


# Adapted policy from Kim, E., Lee, K., & Yoo, C. (2023). Network SLO-aware container scheduling in Kubernetes. The Journal of Supercomputing, 79(10), 11478-11494.
# Original policy deploy a microservice on the node where there are microservices with which the microservice has the greatest traffic exchange
# Our modified policy, for offloading, chose as candidate microservice to move to the edge the one that exchange more traffic with those microservice already at the edge; for unoffloading, remove all micro from the edge and re-add them one at a time as long as the delay increase is below a target value. 
# 

# Set up logger
logger = logging.getLogger('TA_logger')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s SBMP offload %(levelname)s %(message)s'))
logger.propagate = False

def TA_heuristic(params):

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


    # DEFINE DICTIONARY FOR INTERACTION AWARE MATRIX ##
    best = {
        "ms_i": 0,
        "ms_j": 0,
        "interaction_freq": -1
    }

    if delay_decrease_target > 0:
        ## OFFLOAD ##
        while delay_decrease_target > delay_decrease_new:
            ms_candidates = np.argwhere(S_b_new[M:]==0).flatten() # ms not at the edge are candidates for offloading
            ms_edge = np.argwhere(S_b_new[M:]==1).flatten()
            candidate_ms = None
            max_traffic = 0
            for msne in ms_candidates:
                traffic_msne_mse_recv = np.multiply(N[msne]*(Fm[msne, ms_edge].flatten()),L[ms_edge])
                traffic_msne_mse_recv = np.sum(traffic_msne_mse_recv)
                traffic_msne_mse_snt = np.multiply(N[ms_edge],(Fm[ms_edge, msne].flatten()))
                traffic_msne_mse_snt = traffic_msne_mse_snt*L[msne]
                traffic_msne_mse_snt = np.sum(traffic_msne_mse_snt)
                traffic_msne_mse = traffic_msne_mse_recv + traffic_msne_mse_snt
               
                if traffic_msne_mse > max_traffic:
                    max_traffic = traffic_msne_mse
                    candidate_ms = msne
                
            S_b_new[candidate_ms+M] = 1
            
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0] 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                # all instances at the edge
                break
    
    ## UNOFFLOAD  ##
    else:
        ms_origin_edge = np.argwhere(S_b_new[M:2*M-1]==1).flatten()  # microservices at the edge 
        S_b_new = S_b_old.copy()
        S_b_new[M:2*M-1] = 0  # remove all instances at the edge
        Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
        N_new = computeN(Fi_new, M, 2)
        delay_void = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0]
        delay_target = delay_old + delay_increase_target
        delay_new = delay_void
        while delay_new > delay_target:
            # extract the microservices not at the edge
            ms_edge = np.argwhere(S_b_new[M:]==1).flatten()
            ms_candidates = np.setdiff1d(ms_origin_edge, ms_edge)
            max_traffic = 0
            for msne in ms_candidates:
                traffic_msne_mse_recv = np.multiply(N[msne]*(Fm[msne, ms_edge].flatten()),L[ms_edge])
                traffic_msne_mse_recv = np.sum(traffic_msne_mse_recv)
                traffic_msne_mse_snt = np.multiply(N[ms_edge],(Fm[ms_edge, msne].flatten()))
                traffic_msne_mse_snt = traffic_msne_mse_snt*L[msne]
                traffic_msne_mse_snt = np.sum(traffic_msne_mse_snt)
                traffic_msne_mse = traffic_msne_mse_recv + traffic_msne_mse_snt
               
                if traffic_msne_mse > max_traffic:
                    max_traffic = traffic_msne_mse
                    candidate_ms = msne
                
            S_b_new[candidate_ms+M] = 1
            
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)[0] 
            if np.all(S_b_new[M:] == 1):
                # all instances at the edge
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
    result_metrics['cost_increase'] = cost_increase_new
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



