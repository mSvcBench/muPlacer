from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import utils
import logging
import sys
from numpy import inf
from computeNc import computeNc
from buildFci import buildFci
from computeDTot import computeDTot


# Set up logger
logger = logging.getLogger('MFU_heuristic')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s MFU heuristic %(levelname)s %(message)s'))
logger.propagate = False

def mfu_heuristic(params):
    ## VARIABLES INITIALIZATION ##
        
    S_edge_old = params['S_edge_b']
    Acpu_old = params['Acpu']
    Amem_old = params['Amem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target'] if params['mode'] == 'offload' else 0
    delay_increase_target = params['delay_increase_target'] if params['mode'] == 'unoffload' else 0
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']
    Cost_cpu_cloud = params['Cost_cpu_cloud']
    Cost_mem_cloud = params['Cost_mem_cloud']
    Cost_network = params['Cost_network']
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    locked_b = params['locked_b'] if 'locked_b' in params else np.zeros(M)

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    

    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))
    Nci_old = computeNc(Fci_old, M, 2)
    delay_old,_,_,rhoce_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)
    Cost_old = utils.computeCost(Acpu_old, Amem_old, Qcpu, Qmem , Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud,rhoce_old*Ne,Cost_network)[0] # Total cost of old state
    Nc = computeNc(Fcm, M, 1) 

    delay_decrease_new = 0
    delay_increase_new = 0
    S_b_new = S_b_old.copy()
    
    ## OFFLOAD ##
    if params['mode'] == 'offload':
        while delay_decrease_target > delay_decrease_new:
            Nc_max=-1
            argmax = -1
            for i in range(M-1):
                 if Nc[i]>Nc_max and S_b_new[i+M]==0 and locked_b[i]==0:
                    argmax = i
                    Nc_max = Nc[i]
            S_b_new[argmax+M] = 1
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delay_new,_,_,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M) 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                break
        
    ## UNOFFLOAD  ##
    else:
        delay_target = delay_old + delay_increase_target
        
        S_edge_void = np.zeros(int(M))  # (M,) edge state with no instance-set in the edge
        S_edge_void[M-1] = 1  # edge istio proxy
        S_cloud_void = np.ones(int(M))
        S_cloud_void[M-1] = 0
        S_b_void = np.concatenate((S_edge_void, S_edge_void)) # (2*M,) state with no instance-set in the edge

        Acpu_void = np.zeros(2*M)
        Amem_void = np.zeros(2*M)
        Acpu_void[:M] = Acpu_old[:M]+Acpu_old[M:]
        Acpu_void[M:] = np.zeros(M)
        Amem_void[:M] = Amem_old[:M]+Amem_old[M:]
        Amem_void[M:] = np.zeros(M)

        Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
        Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
        delay_void,_,_,rhoce_void = computeDTot(S_b_void, Nci_void, Fci_void, Di, Rs, RTT, Ne, lambd, M)
        delay_decrease_target = max(delay_void - delay_target,0)
        locked_b = np.zeros(M)  # locked microservices binary encoding. 1 if the microservice is locked, 0 otherwise
        locked_b[np.argwhere(S_edge_old==0)] = 1 # microservices that originally where not in the edge are locked
        params = {
            'S_edge_b': S_edge_void.copy(),
            'Acpu': Acpu_void.copy(),
            'Amem': Amem_void.copy(),
            'Qcpu': Qcpu,
            'Qmem': Qmem,
            'Fcm': Fcm.copy(),
            'M': M,
            'lambd': lambd,
            'Rs': Rs[:M],
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'Ne': Ne,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'locked_b': locked_b,
            'mode': 'offload'
        }
        result = mfu_heuristic(params)
        S_b_new = np.ones(2*M)
        S_b_new[M:] = result['S_edge_b']
        

    Acpu_new = np.zeros(2*M)
    Amem_new = np.zeros(2*M)
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)            
    delay_decrease_new = delay_old - delay_new
    np.copyto(Acpu_new,Acpu_old) 
    np.copyto(Amem_new,Amem_old)
    utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old) 
    Cost_new, Cost_new_edge,Cost_cpu_new_edge,Cost_mem_new_edge, Cost_new_cloud,Cost_cpu_new_cloud,Cost_mem_new_cloud,Cost_traffic_new = utils.computeCost(Acpu_new, Amem_new, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_new * Ne, Cost_network) # Total cost of new state
    cost_increase_new = Cost_new - Cost_old 


    # compute final values
    delay_decrease_new = delay_old - delay_new
    delay_increase_new = delay_new - delay_old
    np.copyto(Acpu_new,Acpu_old) 
    np.copyto(Amem_new,Amem_old)
    utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old) 
    Cost_new, Cost_new_edge,Cost_cpu_new_edge,Cost_mem_new_edge, Cost_new_cloud,Cost_cpu_new_cloud,Cost_mem_new_cloud,Cost_traffic_new = utils.computeCost(Acpu_new, Amem_new, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_new * Ne, Cost_network) # Total cost of new state
    cost_increase_new = Cost_new - Cost_old
    cost_decrease_new = Cost_old - Cost_new 

    result_edge = dict()
    
    # extra information
    result_edge['S_edge_b'] = S_b_new[M:].astype(int)
    result_edge['Cost'] = Cost_new
    result_edge['Cost_edge'] = Cost_new_edge
    result_edge['Cost_cpu_edge'] = Cost_cpu_new_edge
    result_edge['Cost_mem_edge'] = Cost_mem_new_edge
    result_edge['Cost_cloud'] = Cost_new_cloud
    result_edge['Cost_cpu_cloud'] = Cost_cpu_new_cloud
    result_edge['Cost_mem_cloud'] = Cost_mem_new_cloud
    result_edge['Cost_traffic'] = Cost_traffic_new
    result_edge['delay_decrease'] = delay_decrease_new
    result_edge['cost_increase'] = cost_increase_new
    result_edge['Acpu'] = Acpu_new
    result_edge['Amem'] = Amem_new
    result_edge['Fci'] = Fci_new
    result_edge['Nci'] = Nci_new
    result_edge['delay'] = delay_new
    result_edge['di'] = di_new
    result_edge['dn'] = dn_new
    result_edge['rhoce'] = rhoce_new
    
    # required return information
     
    result_cloud = dict()
    result_cloud['to-apply'] = list()
    result_cloud['to-delete'] = list()
    result_cloud['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[:M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"


    result_edge['to-apply'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]-S_b_old[M:]>0))
    result_edge['to-delete'] = utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-S_b_new[M:]>0))
    result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_edge['delay_decrease'] < delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    return result_return

    return result