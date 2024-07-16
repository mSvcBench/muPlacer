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
    #delay_decrease_target delay reduction target
    #RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    #Ne cloud-edge network bitrate
    #Cost_cpu_edge cost of CPU at the edge
    #Cost_mem_edge cost of Memory at the edge
    #Di (2*M,) vector of delay of the instance-set at the cloud (:M) and at the edge (M:)
    #Qmem (2*M,) vector of memory quota of the instance at the cloud (:M) and at the edge (M:)
    #Qcpu (2*M,) vector of CPU quota of the instance at the cloud (:M) and at the edge (M:)
    #dependency_paths_b (N,M) binary-based (b) pre computed dependency paths
    #S_edge_base_b (M,) binary-based (b) vector indicating the base edge microservices that can not be removed
    #look_ahead factor to increase the delay reduction target to allow more pruning


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
    dependency_paths_b = params['dependency_paths_b'] if 'dependency_paths_b' in params else None
    S_edge_base_b = params['S_edge_base_b'] if 'S_edge_base_b' in params else np.zeros(M)
    S_edge_base_b[M-1] = 1 # user/ingress  at the edge
    look_ahead = params['look_ahead'] if 'look_ahead' in params else 1.3

    
    S_cloud_old = np.ones(int(M)) # EPAMP assumes all microservice instance run in the cloud
    S_cloud_old[M-1] = 0  # # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1
    S_b_old = np.concatenate((S_cloud_old, S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)[0]  # Total delay of the current configuration. It includes only network delays
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the current state

    ## BUILDING OF DEPENDENCY PATHS ##
    if dependency_paths_b is None:
        G = nx.DiGraph(Fcm) # Create microservice dependency graph 
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

        ## COMPUTE DEPENDENCY PATHS WITH ALL MICROSERIVES AT THE EDGE##
        for ms in range(M-1):
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms))
            for path_n in paths_n:
                # path_n numerical id (n) of the microservices of the dependency path
                # If not all microservices in the path are in the edge this path is not a edge-only
                if not all(S_b_old[M+np.array([path_n])].squeeze()==1):
                    continue
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    
    logger.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_new = S_b_old.copy()
    S_b_opt = S_b_new.copy()
    S_b_temp = np.zeros(2*M)
    Acpu_new = Acpu_old.copy()
    Amem_new = Amem_old.copy()
    while True:
        dpi_best = -1 # index of the leaf microservice to remove
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
        Nci_new = computeNc(Fci_new, M, 2)
        delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
        utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
        Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
        edge_leaves = np.logical_and(np.sum(Fci_new[M:2*M-1,M:2*M-1], axis=1)==0, S_b_new[M:2*M-1].reshape((M-1,1))==1) # edge microservice with no outgoing calls
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        
        # dependency paths that contain current leaves
        dp_with_edges = np.argwhere(dependency_paths_b[:,edge_leaves].sum(axis=1)>0)[:,0]
        
        #logger.info(f'pruning for leaves {edge_leaves}')
        delay_increase_v = np.zeros((len(dp_with_edges),2))   # delay reduction and weigth for each possible removal
        np.copyto(S_b_temp,S_b_new)
        for dpi,path_b in enumerate(dependency_paths_b[dp_with_edges]):
            dependency_paths_b_temp = np.delete(dependency_paths_b,dp_with_edges[dpi],axis=0)
            
            S_b_temp[M:] = np.minimum(np.sum(dependency_paths_b_temp,axis=0),1)
            S_b_temp[M:] = S_b_temp[M:]+S_edge_base_b
            S_b_temp[S_b_temp>0] = 1
            Acpu_temp = np.zeros(2*M)
            Amem_temp = np.zeros(2*M)
            Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))
            Nci_temp = computeNc(Fci_temp, M, 2)
            delay_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
            delay_increase_temp = max(1e-6,delay_temp - delay_new)
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            Cost_edge_temp = utils.computeCost(Acpu_temp[M:], Amem_temp[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
            cost_decrease = Cost_edge_new - Cost_edge_temp
            w = cost_decrease/delay_increase_temp
            delay_increase_v[dpi] = [delay_increase_temp,w]
        
        feasible_dpi = np.argwhere(delay_increase_v[:,0]<=delay_increase_target).flatten()
        if len(feasible_dpi)>0:
            dpi_best = np.argmax(delay_increase_v[feasible_dpi][:,1])
            dpi_best = feasible_dpi[dpi_best]
        else:
            feasible_dpi = np.argwhere(delay_increase_v[:,0]<delay_increase_target * look_ahead).flatten()
            if len(feasible_dpi)>0:
                dpi_best = np.argmax(delay_increase_v[feasible_dpi][:,1])
                dpi_best = feasible_dpi[dpi_best]
        
        if dpi_best>-1:
            delay_increase_best = delay_increase_v[dpi_best,0]
            logger.info(f'cleaning dependency path {np.argwhere(dependency_paths_b[dp_with_edges[dpi_best]]).squeeze()}, delay increase: {delay_increase_best}')
            dependency_paths_b = np.delete(dependency_paths_b,dp_with_edges[dpi_best],axis=0)
            S_b_new[M:] = np.minimum(np.sum(dependency_paths_b,axis=0),1)
            S_b_new[M:] = S_b_new[M:]+S_edge_base_b
            S_b_new[S_b_new>0] = 1
            if delay_increase_best <= delay_increase_target:
                np.copyto(S_b_opt,S_b_new)
        else:
            break
            
    logger.info(f"++++++++++++++++++++++++++++++")
    np.copyto(S_b_new,S_b_opt)
    
    # compute final values
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, np.empty(0))
    delay_increase_new = delay_new - delay_old
    utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
    Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
    cost_decrease_new = Cost_edge_old - Cost_edge_new

    result_edge = dict()
    
    # extra information
    result_edge['S_edge_b'] = S_b_new[M:].astype(int)
    result_edge['Cost'] = Cost_edge_new
    result_edge['delay_increase'] = delay_increase_new
    result_edge['cost_decrease'] = cost_decrease_new
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
    result_cloud['info'] = f"Result for unoffload - cloud microservice ids: {result_cloud['placement']}"


    result_edge['to-apply'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]-S_b_old[M:]>0))
    result_edge['to-delete'] = utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-S_b_new[M:]>0))
    result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[M:]==1))

    result_edge['info'] = f"Result for unoffload - edge microservice ids: {result_edge['placement']}"
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    return result_return
    
