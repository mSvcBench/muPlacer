from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import utils
import logging
import sys
from numpy import inf
from computeN import computeN
from buildFi import buildFi
from computeDTot import computeDTot


# Set up logger
logger = logging.getLogger('MFU_heuristic')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s MFU heuristic %(levelname)s %(message)s'))
logger.propagate = False

def mfu(params):
    ## VARIABLES INITIALIZATION ##
        
    S_edge_old = params['S_edge_b']
    Ucpu_old = params['Ucpu']
    Umem_old = params['Umem']
    Fm = params['Fm']
    M = params['M']
    lambd = params['lambd']
    L = params['L']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target'] if params['mode'] == 'offload' else 0
    delay_increase_target = params['delay_increase_target'] if params['mode'] == 'unoffload' else 0
    RTT = params['RTT']
    B = params['B']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']
    Cost_cpu_cloud = params['Cost_cpu_cloud']
    Cost_mem_cloud = params['Cost_mem_cloud']
    Cost_network = params['Cost_network']
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    global_HPA_cpu_th = params['global_HPA_cpu_th'] if 'global_HPA_cpu_th' in params else None
    locked_b = params['locked_b'] if 'locked_b' in params else np.zeros(M)

    L = np.tile(L, 2)  # Expand the Rs vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    

    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fi_old = np.matrix(buildFi(S_b_old, Fm, M))
    N_old = computeN(Fi_old, M, 2)
    delay_old,_,_,rhoce_old = computeDTot(S_b_old, N_old, Fi_old, Di, L, RTT, B, lambd, M)
    Cost_old = utils.computeCost(Ucpu_old, Umem_old, Qcpu, Qmem , Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud,rhoce_old*B,Cost_network, global_HPA_cpu_th)[0] # Total cost of old state
    N = computeN(Fm, M, 1) 

    delay_decrease_new = 0
    delay_increase_new = 0
    S_b_new = S_b_old.copy()
    
    ## OFFLOAD ##
    if params['mode'] == 'offload':
        while delay_decrease_target > delay_decrease_new:
            N_max=-1
            argmax = -1
            for i in range(M-1):
                 if N[i]>N_max and S_b_new[i+M]==0 and locked_b[i]==0:
                    argmax = i
                    N_max = N[i]
            S_b_new[argmax+M] = 1
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new,_,_,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M) 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                break
        
    ## UNOFFLOAD  ##
    else:
        while True:
            N_min=np.inf
            argmax = -1
            for i in range(M-1):
                if N[i]<N_min and S_b_new[i+M]==1:
                    argmax = i
                    N_min = N[i]
            S_b_new[argmax+M] = 0
            Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
            N_new = computeN(Fi_new, M, 2)
            delay_new,_,_,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M) 
            delay_increase_new = delay_new-delay_old
            if np.all(S_b_new[M:] == 0):
                break
            if delay_increase_new > delay_increase_target:
                S_b_new[argmax+M] = 1
                break
        
        # S_edge_void = np.zeros(int(M))  # (M,) edge state with no instance-set in the edge
        # S_edge_void[M-1] = 1  # edge istio proxy
        # S_cloud_void = np.ones(int(M))
        # S_cloud_void[M-1] = 0
        # S_b_void = np.concatenate((S_edge_void, S_edge_void)) # (2*M,) state with no instance-set in the edge

        # Ucpu_void = np.zeros(2*M)
        # Umem_void = np.zeros(2*M)
        # Ucpu_void[:M] = Ucpu_old[:M]+Ucpu_old[M:]
        # Ucpu_void[M:] = np.zeros(M)
        # Umem_void[:M] = Umem_old[:M]+Umem_old[M:]
        # Umem_void[M:] = np.zeros(M)

        # Fi_void = np.matrix(buildFi(S_b_void, Fm, M))    # instance-set call frequency matrix of the void state
        # N_void = computeN(Fi_void, M, 2)    # number of instance call per user request of the void state
        # delay_void,_,_,rhoce_void = computeDTot(S_b_void, N_void, Fi_void, Di, L, RTT, B, lambd, M)
        # delay_decrease_target = max(delay_void - delay_target,0)
        # locked_b = np.zeros(M)  # locked microservices binary encoding. 1 if the microservice is locked, 0 otherwise
        # locked_b[np.argwhere(S_edge_old==0)] = 1 # microservices that originally where not in the edge are locked
        # params = {
        #     'S_edge_b': S_edge_void.copy(),
        #     'Ucpu': Ucpu_void.copy(),
        #     'Umem': Umem_void.copy(),
        #     'Qcpu': Qcpu,
        #     'Qmem': Qmem,
        #     'Fm': Fm.copy(),
        #     'M': M,
        #     'lambd': lambd,
        #     'L': L[:M],
        #     'Di': Di,
        #     'delay_decrease_target': delay_decrease_target,
        #     'RTT': RTT,
        #     'B': B,
        #     'Cost_cpu_edge': Cost_cpu_edge,
        #     'Cost_mem_edge': Cost_mem_edge,
        #     'Cost_cpu_cloud': Cost_cpu_cloud,
        #     'Cost_mem_cloud': Cost_mem_cloud,
        #     'Cost_network': Cost_network,
        #     'locked_b': locked_b,
        #     'mode': 'offload'
        # }
        # result = mfu(params)
        # S_b_new = np.ones(2*M)
        # S_b_new[M:] = result['S_edge_b']
        

    # compute final values
    Ucpu_new = np.zeros(2*M)
    Umem_new = np.zeros(2*M)
    Fi_new = np.matrix(buildFi(S_b_new, Fm, M))
    N_new = computeN(Fi_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, Di, L, RTT, B, lambd, M)            
    delay_decrease_new = delay_old - delay_new
    delay_increase_new = delay_new - delay_old
    np.copyto(Ucpu_new,Ucpu_old) 
    np.copyto(Umem_new,Umem_old)
    utils.computeResourceShift(Ucpu_new, Umem_new, N_new, Ucpu_old, Umem_old, N_old) 
    Cost_new, Cost_new_edge, Cost_new_cloud, Cost_traffic_new = utils.computeCost(Ucpu_new, Umem_new, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_new * B, Cost_network, global_HPA_cpu_th) # Total cost of new state
    cost_increase_new = Cost_new - Cost_old 


    # # compute final values
    # delay_decrease_new = delay_old - delay_new
    # delay_increase_new = delay_new - delay_old
    # np.copyto(Acpu_new,Acpu_old) 
    # np.copyto(Amem_new,Amem_old)
    # utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old) 
    # Cost_new, Cost_new_edge,Cost_cpu_new_edge,Cost_mem_new_edge, Cost_new_cloud,Cost_cpu_new_cloud,Cost_mem_new_cloud,Cost_traffic_new = utils.computeCost(Acpu_new, Amem_new, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_new * Ne, Cost_network) # Total cost of new state
    # cost_increase_new = Cost_new - Cost_old
    # cost_decrease_new = Cost_old - Cost_new 

    result_metrics = dict()
    
    # extra information
    result_metrics['S_edge_b'] = S_b_new[M:].astype(int)
    result_metrics['Cost'] = Cost_new
    result_metrics['Cost_edge'] = Cost_new_edge
    result_metrics['Cost_cloud'] = Cost_new_cloud
    result_metrics['Cost_traffic'] = Cost_traffic_new
    result_metrics['delay_decrease'] = delay_decrease_new
    result_metrics['delay_increase'] = delay_increase_new
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
    result_cloud['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[:M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"

    result_edge = dict()
    result_edge['to-apply'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]-S_b_old[M:]>0))
    result_edge['to-delete'] = utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-S_b_new[M:]>0))
    result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_metrics['delay_decrease'] < delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    result_return.append(result_metrics)
    return result_return