# pylint: disable=C0103, C0301

import logging
import sys
import argparse
import numpy as np
import networkx as nx
import utils
from numpy import inf
from computeN import computeN
from buildFi import buildFi
from computeDTot import computeDTot
from SBMP_offload import sbmp_o


np.seterr(divide='ignore', invalid='ignore')
# Set up logger
logger = logging.getLogger('SBMP_unoffload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP unoffload %(levelname)s %(message)s'))
logger.propagate = False

def sbmp_u(params):

    ## INITIALIZE VARIABLES ##

    # mandatory paramenters
    S_edge_old = params['S_edge_b'] # binary vector indicating if the microservice is at the edge or not
    Ucpu_old = params['Ucpu'] # Used CPU per microservice
    Umem_old = params['Umem'] # Used memory per microservice
    Fm = params['Fm'] # microservice call frequency matrix
    M = params['M'] # number of microservices
    lambd = params['lambd'] # user request rate
    L = params['L'] # response size of microservices
    delay_increase_target = params['delay_increase_target'] # requested delay increase
    delay_increase_stop_condition = params['delay_increase_stop_condition'] if 'delay_increase_stop_condition' in params else delay_increase_target # stop condition for delay increase
    RTT = params['RTT'] # fixed delay to add to microservice interaction in addition to the time depending on the response size
    B = params['B'] # network bitrate cloud-edge
    Cost_cpu_edge = params['Cost_cpu_edge'] # Cost of CPU at the edge per hour
    Cost_mem_edge = params['Cost_mem_edge'] # Cost of 1 GB memory at the edge per hour
    Cost_cpu_cloud = params['Cost_cpu_cloud'] # Cost of CPU at the cloud per hour
    Cost_mem_cloud = params['Cost_mem_cloud'] # Cost of 1 GB memory at the cloud per hour
    Cost_network = params['Cost_network']   # Cost of network per GB
    
    # optional paramenters
    Di = params['Di'] if 'Di' in params else np.zeros(2*M) # internal delay of microservices, 0 for homogeneous hardware
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M) # memory quota per microservice instance (Kubernetes Request)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M) # CPU quota per microservice instance (Kubernetes Request)
    cache_ttl = params['cache-ttl'] if 'cache-ttl' in params else 10 # cache expiry in round
    expanding_depth = params['expanding-depth'] if 'expanding-depth' in params else M # maximum number of microservices upgrade to consider in the single path adding greedy iteraction (lower reduce optimality but increase computaiton speed)
    traces_b = params['traces-b'] if 'traces-b' in params else None # flag to enable traces generation
    max_sgs = params['max-sgs'] if 'max-sgs' in params else 1e6 # maximum number of subgraphs to consider in an optimization iteration
    max_traces = params['max-traces'] if 'max-traces' in params else 1024 # maximum number of traces to generate
    HPA_cpu_th = params['HPA_cpu_th'] if 'HPA_cpu_th' in params else None # CPU threshold for HPA
    
    S_cloud_old = np.ones(int(M)) # SBMP assumes all microservice instance run in the cloud
    S_cloud_old[M-1] = 0  # # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1
    S_b_old = np.concatenate((S_cloud_old, S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge

    L = np.tile(L, 2)  # Expand the Rs vector to support matrix operations
    Fi_old = np.matrix(buildFi(S_b_old, Fm, M)) # (2*M,2*M) instance-set call frequency matrix
    Ni_old = computeN(Fi_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old,_,_,rhoce_old = computeDTot(S_b_old, Ni_old, Fi_old, Di, L, RTT, B, lambd, M)  # Total delay of the current configuration. It includes only network delays
    delay_target = delay_old + delay_increase_target # Target delay to reach 
    delay_stop = delay_old + delay_increase_stop_condition # Stop condition for delay increase
    Cost_old = utils.computeCost(Ucpu_old, Umem_old, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_old * B, Cost_network)[0] # Total cost of the current state

    S_edge_void = np.zeros(int(M))  # (M,) edge state with no instance-set in the edge
    S_edge_void[M-1] = 1  # edge istio proxy (user)
    S_cloud_void = np.ones(int(M)) # (M,) cloud state with all microservices in the cloud
    S_cloud_void[M-1] = 0 # user not in the cloud
    S_b_void = np.concatenate((S_edge_void, S_edge_void)) # (2*M,) cloud-only state, void edge

    Ucpu_void = np.zeros(2*M)
    Umem_void = np.zeros(2*M)
    Ucpu_void[:M] = Ucpu_old[:M]+Ucpu_old[M:]
    Ucpu_void[M:] = np.zeros(M)
    Umem_void[:M] = Umem_old[:M]+Umem_old[M:]
    Umem_void[M:] = np.zeros(M)

    Fi_void = np.matrix(buildFi(S_b_void, Fm, M))    # instance-set call frequency matrix of the void state
    N_void = computeN(Fi_void, M, 2)    # number of instance call per user request of the void state
    delay_void = computeDTot(S_b_void, N_void, Fi_void, Di, L, RTT, B, lambd, M)[0]
    delay_decrease_target = max(delay_void - delay_target,0) # target delay decrease from void state
    delay_decrease_stop_condition = max(delay_void - delay_stop,0) # stop condition for delay decrease
    locked_b = np.zeros(M)  # locked microservices binary encoding. 1 if the microservice is locked, 0 otherwise
    locked_b[np.argwhere(S_edge_old==0)] = 1 # microservices that originally where not in the edge are locked
    params = {
        'S_edge_b': S_edge_void.copy(),
        'Ucpu': Ucpu_void.copy(),
        'Umem': Umem_void.copy(),
        'Qcpu': Qcpu,
        'Qmem': Qmem,
        'Fm': Fm.copy(),
        'M': M,
        'lambd': lambd,
        'L': L[:M],
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'delay_decrease_stop_condition': delay_decrease_stop_condition,
        'RTT': RTT,
        'B': B,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'Cost_cpu_cloud': Cost_cpu_cloud,
        'Cost_mem_cloud': Cost_mem_cloud,
        'Cost_network': Cost_network,
        'locked_b': locked_b,
        'sgs-builder': 'sgs_builder_traces',
        'max-sgs': max_sgs,
        'max-traces': max_traces,
        'traces-b': traces_b,
        'mode': 'unoffload',
        'HPA_cpu_th': HPA_cpu_th,
        'expanding_depth': expanding_depth,
        'cache-ttl': cache_ttl
    }
    logger.info(f"unoffload calls offload with void edge and delay_decrease_target: {delay_decrease_target}")
    result_list = sbmp_o(params)
    result=result_list[2]
    result['delay_increase'] = (delay_void-result['delay_decrease']) - delay_old
    result['cost_decrease'] = Cost_old-result['Cost']
    result['to-apply'] = utils.numpy_array_to_list(np.argwhere(result['S_edge_b']-S_b_old[M:]>0))
    result['to-delete']= utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-result['S_edge_b']>0))
    del result['delay_decrease']
    del result['cost_increase']
    message = f"Result for unoffload - edge microservice ids: {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}"
    result['info'] = message
    return result_list
