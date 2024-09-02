# pylint: disable=C0103, C0301

import logging
import sys
import argparse
import numpy as np
import networkx as nx
import utils
from numpy import inf
from computeNc import computeNc
from buildFci import buildFci
from computeDTot import computeDTot
from EPAMP_offload_sweeping import offload


np.seterr(divide='ignore', invalid='ignore')
# Set up logger
logger = logging.getLogger('EPAMP_unoffload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP unoffload %(levelname)s %(message)s'))
logger.propagate = False

def unoffload(params):

    ## INITIALIZE VARIABLES ##
    #Acpu_old (2*M,) vector of CPU req by instance-set at the cloud (:M) and at the edge (M:)
    #Amem_old (2*M,) vector of Memory req by instance-set at the cloud (:M) and at the edge (M:)
    #Fcm (M,M)microservice call frequency matrix
    #M number of microservices
    #lambd user request rate
    #Rs (M,) vector of response size of microservices
    #S_edge_old (M,) vector of binary values indicating if the microservice is at the edge or not
    #delay_increase_target delay increase target
    #RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    #Ne cloud-edge network bitrate
    #Cost_cpu_edge cost of CPU at the edge
    #Cost_mem_edge cost of Memory at the edge
    #Di (2*M,) vector of internal delay of an instance at the cloud (:M) and at the edge (M:)
    # Qmem (M,) memory quantum in bytes, Kubernetes memory request
    # Qcpu (M,) CPU quantum in cpu sec, Kubernetes CPU request

    # mandatory paramenters
    S_edge_old = params['S_edge_b']
    Acpu_old = params['Acpu']
    Amem_old = params['Amem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    delay_increase_target = params['delay_increase_target']
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']

    
    # optional paramenters
    Di = params['Di'] if 'Di' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    
    S_cloud_old = np.ones(int(M)) # EPAMP assumes all microservice instance run in the cloud
    S_cloud_old[M-1] = 0  # # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1
    S_b_old = np.concatenate((S_cloud_old, S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)[0]  # Total delay of the current configuration. It includes only network delays
    delay_target = delay_old + delay_increase_target
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the current state

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
    delay_void = computeDTot(S_b_void, Nci_void, Fci_void, Di, Rs, RTT, Ne, lambd, M)[0]
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
        'locked_b': locked_b
    }
    logger.info(f"unoffload calls offload with void edge and delay_decrease_target: {delay_decrease_target}")
    result_list = offload(params)
    result=result_list[1]
    result['delay_increase'] = (delay_void-result['delay_decrease']) - delay_old
    result['cost_decrease'] = Cost_edge_old-result['Cost']
    result['to-apply'] = utils.numpy_array_to_list(np.argwhere(result['S_edge_b']-S_b_old[M:]>0))
    result['to-delete']= utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-result['S_edge_b']>0))
    del result['delay_decrease']
    del result['cost_increase']
    message = f"Result for unoffload - edge microservice ids: {result['placement']}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}"
    result['info'] = message
    return result_list
