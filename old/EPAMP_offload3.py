# pylint: disable=C0103, C0301
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import networkx as nx
from computeNc import computeN
from buildFi import buildFi
from numpy import inf
from computeDTot import computeDTot
import logging
import sys
import utils


np.seterr(divide='ignore', invalid='ignore')

# Set up logger
logger = logging.getLogger('EPAMP_offload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP offload %(levelname)s %(message)s'))
logger.propagate = False

def offload(params):

    ## INITIALIZE VARIABLES ##
    # S_edge_old (M,) vector of binary values indicating if the microservice instance-set is running in the edge or not
    # Acpu_old (2*M,) vector of actual CPU req by instance-set at the cloud (:M) and at the edge (M:)
    # Amem_old (2*M,) vector of actual Memory req by instance-set at the cloud (:M) and at the edge (M:)
    # Fcm(M,M) microservice call frequency matrix
    # M number of microservices
    # lambd user request rate
    # Rs(M,) vector of response size of microservices
    # Di(M,) vector of internal delay of microservices
    # delay_decrease_target delay reduction target
    # RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    # Ne cloud-edge network bitrate
    # Cost_cpu_edge cost of CPU unit at the edge
    # Cost_mem_edge cost of Memory unit at the edge
    # locked(M,) vector of binary values indicating  microservices that can not change state
    # u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)
    # no_evolutionary if True, disable the removal of microservices from the edge to reduce cost
    # max_added_dp maximum number of dependency path added to the current configuration before stopping the greedy iteration
    # min_added_dp minimum number of dependency path added to the current configuration before stopping the greedy iteration
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
    delay_decrease_target = params['delay_decrease_target']
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']

    
    # optional paramenters
    Di = params['Di'] if 'Di' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    max_added_dp = params['max_added_dp'] if 'max_added_dp' in params else 1000000
    min_added_dp = params['min_added_dp'] if 'min_added_dp' in params else 0
    dependency_paths_b = params['dependency_paths_b'] if 'dependency_paths_b' in params else None
    locked = params['locked'] if 'locked' in params else None
    u_limit = params['u_limit'] if 'u_limit' in params else M
    no_evolutionary = params['no_evolutionary'] if 'no_evolutionary' in params else False


    S_cloud_old = np.ones(int(M)) # EPAMP assumes all microservice instance run in the cloud
    S_cloud_old[M-1] = 0 # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1 
    S_b_old = np.concatenate((S_cloud_old, S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    
    G = nx.DiGraph(Fcm) # Create microservice dependency graph
    if nx.is_directed_acyclic_graph(G)==False: # Check if the graph is acyclic
        logger.critical(f"Microservice dependency graph is not acyclic, EPAMP optimization can not be used")
        result_edge=dict()
        result_edge['S_edge_b'] = S_b_old[M:].astype(int)
        result_edge['to-apply'] = list()
        result_edge['to-delete'] = list()
        result_edge['placement'] = numpy_array_to_list(np.argwhere(S_b_old[M:]==1))
        result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}, Cost: {result_edge['Cost']}, delay decrease: {result_edge['delay_decrease']}, cost increase: {result_edge['cost_increase']}"
        return result_edge 
    
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    
    # SAVE INITIAL (OLD) METRICS VALUES ##
    Fci_old = np.matrix(buildFi(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeN(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]  # Total delay of the current configuration. It includes only network delays
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
    

    ## FULL ADDITION OF CLOUD-EDGE DEPENDECY PATHS TO EDGE CLUSTER ##
    S_b_new = S_b_old.copy() # S_b_new is the new placement state 
    S_b_new[M:] = 1 # all microservices are moved to the edge
    Acpu_new = np.zeros(2*M)    # Acpu_new is the new CPU request vector
    Amem_new = np.zeros(2*M)    # Amem_new is the new Memory request vector
    Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))    # instance-set call frequency matrix of the temp state
    Nci_new = computeN(Fci_new, M, 2)  
    utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
    
    ## BUILDING OF DEPENDENCY PATHS ##
    if dependency_paths_b is None:
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

        ## COMPUTE DEPENDENCY PATHS WITH SOME MICROSERVICE IN THE CLOUD ##
        for ms in range(M-1):
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            for path_n in paths_n:
                # path_n numerical id (n) of the microservices of the dependency path
                # If all microservices in the path are in the edge this path is not a cloud-edge path
                if all(S_b_old[M+np.array([path_n])].squeeze()==1):
                    continue
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
                    
    logger.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_old_edge = np.array(S_b_old[M:]).reshape(M,1)
    S_b_opt = S_b_new.copy()
    while True:
        w_opt = -1
        c_max = 0 # max cost of the leaf microservice to remove
        dpi_best = -1 # index of the leaf microservice to remove
        S_b_temp = np.zeros(2*M)
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
        Nci_new = computeN(Fci_new, M, 2)
        delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
        utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
        Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
        edge_leaves = np.logical_and(np.sum(Fci_new[M:2*M-1,M:2*M-1], axis=1)==0, S_b_new[M:2*M-1].reshape((M-1,1))==1) # edge microservice with no outgoing calls
        # if (no_evolutionary==False):
        #     edge_leaves = np.logical_and(edge_leaves, S_b_old_edge[M-1]==0)    # old edge microservice can not be removed for incremental constraint
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        
        # dependency paths that contain current leaves
        dp_with_edges = np.argwhere(dependency_paths_b[:,edge_leaves].sum(axis=1)>0)[:,0]
        
        # filter out dependency paths not contained in the current states
        dp_with_edges_valid_id = np.argwhere(np.equal(np.multiply(dependency_paths_b[dp_with_edges],S_b_new[M:]),dependency_paths_b[dp_with_edges]).all(axis=1))[:,0]
        dp_with_edges = dp_with_edges[dp_with_edges_valid_id]
        
        #logger.info(f'pruning for leaves {edge_leaves}')
        S_b_temp = S_b_new.copy()
        for dpi,path_b in enumerate(dependency_paths_b[dp_with_edges]):
            dependency_paths_b_temp = np.delete(dependency_paths_b,dp_with_edges[dpi],axis=0)
            
            S_b_temp[M:] = np.minimum(np.sum(dependency_paths_b_temp,axis=0),1)
            Acpu_temp = np.zeros(2*M)
            Amem_temp = np.zeros(2*M)
            Fci_temp = np.matrix(buildFi(S_b_temp, Fcm, M))
            Nci_temp = computeN(Fci_temp, M, 2)
            delay_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
            delay_increase_temp = max(1e-6,delay_temp - delay_new)
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            Cost_edge_temp = utils.computeCost(Acpu_temp[M:], Amem_temp[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
            cost_decrease = Cost_edge_new - Cost_edge_temp
            w = cost_decrease/delay_increase_temp
            if w>w_opt:
                # possible removal
                w_opt = w
                dpi_best = dpi
                delay_reduction_best = delay_old - delay_temp
        if dpi_best>-1:
            logger.info(f'cleaning dependency path {np.argwhere(dependency_paths_b[dp_with_edges[dpi_best]]).squeeze()}, delay reduction: {delay_reduction_best}')
            dependency_paths_b = np.delete(dependency_paths_b,dp_with_edges[dpi_best],axis=0)
            S_b_new[M:] = np.minimum(np.sum(dependency_paths_b,axis=0),1)
            if delay_reduction_best >= delay_decrease_target:
                np.copyto(S_b_opt,S_b_new)
        else:
            break
            
    logger.info(f"++++++++++++++++++++++++++++++")
    
    # compute final values
    np.copyto(S_b_new,S_b_opt)
    Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
    Nci_new = computeN(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, np.empty(0))
    delay_decrease_new = delay_old - delay_new
    utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
    Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
    cost_increase_new = Cost_edge_new - Cost_edge_old

    result_edge = dict()
    
    # extra information
    result_edge['S_edge_b'] = S_b_new[M:].astype(int)
    result_edge['Cost'] = Cost_edge_new
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

