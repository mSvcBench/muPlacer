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
import strategies.utils as utils
import random
import S2id


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


    if dependency_paths_b is None:
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
        ## BUILDING OF DP_TOT RANDOM DEPENDENCY PATHS ##
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
        DP_TOT = 100000 # Total number of dependency paths to consider
        dp_ids=list()
        for i in range(DP_TOT):
            
            span=set()
            span.add(M-1)
            span_old=set()
            span_old.add(M-1)
            span_new=set()
            while True:
                for j in span_old:
                    for k in np.argwhere(Fcm[j,:]>0).flatten().tolist():
                        if random.random() < Fcm[j,k]:
                            span_new.add(k)
                            span.add(k)
                if len(span_new)>0:
                    span_old=span_new.copy()
                    span_new=set()
                else:
                    break
            path_n = list(span)
            if all(S_b_old[M+np.array([path_n])].squeeze()==1):
                continue
            else:
                path_b = np.zeros((1,M),int)
                path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                if S2id.S2id(path_b[0]) in dp_ids:
                    continue
                dp_ids.append(S2id.S2id(path_b[0]))
                logger.debug(f'new {i} dependency path {np.argwhere(path_b[0]==1).flatten()}')
                dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)

        # ## COMPUTE DEPENDENCY PATHS WITH SOME MICROSERVICE IN THE CLOUD ##
        # for ms in range(M-1):
        #     paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        #     for path_n in paths_n:
        #         # path_n numerical id (n) of the microservices of the dependency path
        #         # If all microservices in the path are in the edge this path is not a cloud-edge path
        #         if all(S_b_old[M+np.array([path_n])].squeeze()==1):
        #             continue
        #         else:
        #             path_b = np.zeros((1,M),int)
        #             path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
        #             dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    

    ## GREEDY ADDITION OF CLOUD-EDGE DEPENDECY PATHS TO EDGE CLUSTER ##
    dependency_paths_b_residual = dependency_paths_b.copy() # residual dependency path to consider in a greedy round, \Pi_r of paper
    S_b_opt = S_b_old.copy()  # S_b_opt is the best placement state computed by a greedy round
    S_b_temp = np.zeros(2*M) # S_b_temp is the temporary placement state used in a greedy round
    S_b_new = np.zeros(2*M) # S_b_new is the new placement state 
    Acpu_opt = Acpu_old.copy()  # Acpu_opt is the best CPU request vector computed by a greedy round
    Amem_opt = Amem_old.copy()  # Amem_opt is the best Memory request vector computed by a greedy round
    Acpu_new = np.zeros(2*M)    # Acpu_new is the new CPU request vector
    Amem_new = np.zeros(2*M)    # Amem_new is the new Memory request vector
    Acpu_temp = np.zeros(2*M)   # Acpu_temp is the temporary CPU request vector used in a greedy round
    Amem_temp = np.zeros(2*M)   # Amem_temp is the temporary Memory request vector used in a greedy round
    delay_opt = delay_old   # delay_opt is the best delay computed by a greedy round. It includes only network delays
    Nci_new = Nci_old.copy()
    Nci_opt = Nci_old.copy()

    skip_delay_increase = False    # Skip delay increase states to accelerate computation wheter possible
    locking = False if locked is None else True # avoid locking control if no microservice is locked
    cost_increase_opt=0    # cost_increase_opt is the best cost increase computed by a greedy round
    delay_decrease_opt=1   # delay_decrease_opt is the best delay reduction computed by a greedy round
    
    if min_added_dp < 0:
        # min_added_dp is the minimum number of dependency path to add before stopping the greedy iteration
        # negative value means that the minimum is equal to the whole set of dependency path minus the input value
        min_added_dp = len(dependency_paths_b_residual) + min_added_dp
    
    logger.info(f"ADDING PHASE")
   
    while True:
        logger.info(f'-----------------------')
        w_min = float("inf") # Initialize the weight
        skip_delay_increase = False    # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)  
        np.copyto(Acpu_new,Acpu_opt)    # Acpu_new is the new CPU request vector, Acpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Amem_new,Amem_opt)    # Amem_new is the new Memory request vector, Amem_opt is the best Memory request vector computed by the previos greedy round
        np.copyto(Nci_new,Nci_opt)
        delay_new = delay_opt   # delay_new is the new delay. It includes only network delays
        Cost_edge_new  = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the new configuration
        logger.info(f'new state {np.argwhere(S_b_new[M:]==1).squeeze()}, delay decrease {1000*(delay_old-delay_new)}, cost {Cost_edge_new}, cost increase / delay decrease {cost_increase_opt/(1000*delay_decrease_opt)}')
        
        # Check if the delay reduction and other constraints are reached
        added_dp = len(dependency_paths_b)-len(dependency_paths_b_residual) # number of dependency path added so far

        if delay_old-delay_new >= delay_decrease_target and added_dp >= min_added_dp:
            #delay reduction reached with minimum number of dependency paths added
            logger.info(f'delay reduction reached with minimum number of dependency paths added')
            break

        if added_dp >= max_added_dp:
            # max number of dependency paths to add reached
            logger.info(f'max number of dependency paths to add reached')
            break

        if len(dependency_paths_b_residual) == 0:
            # All dependency path considered no other way to reduce delay
            logger.info(f'All dependency path considered no other way to reduce delay')
            break

        ## GREEDY ROUND ##
        
        # u_limit complexity reduction
        # for the next greedy round, select dependency paths providing a number of microservice upgrade not greater than u_limit
        logger.debug(f"number of depencency paths without upgrade limit: {len(dependency_paths_b_residual)}")
        rl = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=u_limit)   # index of dependency paths with microservices upgrade less than u_limit
        logger.debug(f"number of depencency paths with upgrade limit: {len(rl)}")
        
        for path_b in dependency_paths_b_residual[rl] :
            # merging path_b and S_b_new into S_b_temp
            path_n = np.argwhere(path_b.flatten()==1).squeeze() # numerical id of the microservices of the dependency path
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[M+path_n] = 1
            
            #check looked microservices
            if locking:
                if not np.equal(S_b_temp[M:]*locked, S_b_old[M:]*locked).all(): # if a locked microservice is moved, skip
                    continue
            
            Fci_temp = np.matrix(buildFi(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
            Nci_temp = computeN(Fci_temp, M, 2)    # number of instance call per user request of the temp state
            delay_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0] # Total delay of the temp state. It includes only network delays
            delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
            if skip_delay_increase and delay_decrease_temp<0:
                logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()} skipped for negative delay decrease')
                continue
            
            # compute the cost increase adding this dependency path 
            # assumption is that cloud resource are reduce proportionally with respect to the reduction of the number of times instances are called
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_old,Amem_old,Nci_old)
            Cost_edge_temp = utils.computeCost(Acpu_temp[M:], Amem_temp[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the temp state
            cost_increase_temp = Cost_edge_temp - Cost_edge_new # cost increase wrt the new state
            
            # weighting
            r_delay_decrease = delay_decrease_target - (delay_old-delay_new) # residul delay to decrease wrt previous conf
            if delay_decrease_temp <= 0:
                # addition provides delay increase,  weighting penalize both cost and delay increase
                # edge_freq_increase = np.sum(Nci_temp[M:])-np.sum(Nci_new[M:])
                # w = 1e6 + cost_increase_temp/edge_freq_increase
                w = 1e6 - cost_increase_temp * 1000 * delay_decrease_temp   # 1000 usedto move delay in the ms scale
            else:
                w = cost_increase_temp /  max(min(1000*delay_decrease_temp, 1000*r_delay_decrease),1e-3) # 1e-3 used to avoid division by zero
                skip_delay_increase = True
            
            logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()}, cost increase {cost_increase_temp},delay decrease {1000*delay_decrease_temp}, delay {delay_temp}, weight {w}')

            if w < w_min:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                np.copyto(Acpu_opt,Acpu_temp)
                np.copyto(Amem_opt,Amem_temp)
                np.copyto(Nci_opt,Nci_temp)
                cost_increase_opt = cost_increase_temp
                delay_decrease_opt = delay_decrease_temp
                delay_opt = delay_temp
                w_min = w
        
        if w_min == inf:
            # no improvement possible in the greedy round
            logger.info(f'no improvement possible in the greedy round')
            break

        # Prune not considered dependency paths whose microservices are going to be contained in the edge to accelerate computation
        PR = []
        for pr,path_b in enumerate(dependency_paths_b_residual):
            if np.sum(path_b) == np.sum(path_b * S_b_opt[M:]):
                # dependency path already fully included at edge
                PR.append(pr)
        dependency_paths_b_residual = np.delete(dependency_paths_b_residual, PR, axis=0)


    logger.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_old_a = np.array(S_b_old[M:]).reshape(M,1)
    while added_dp > min_added_dp:
        c_max = 0 # max cost of the leaf microservice to remove
        leaf_max = -1 # index of the leaf microservice to remove
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
        S_b_new_a = np.array(S_b_new[M:]).reshape(M,1)
        edge_leaves = np.logical_and(np.sum(Fci_new[M:,:], axis=1)==0, S_b_new_a==1) # edge microservice with no outgoing calls
        if (no_evolutionary):
            edge_leaves = np.logical_and(edge_leaves, S_b_old_a==0)    # old edge microservice can not be removed for incremental constraint
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        edge_leaves = edge_leaves+M # index of the edge microservice in the full state
        for leaf in edge_leaves:
            # try remove microservice
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[leaf] = 0
            Fci_temp = np.matrix(buildFi(S_b_temp, Fcm, M))
            Nci_temp = computeN(Fci_temp, M, 2)
            delay_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
            delay_decrease_temp = delay_old - delay_temp
            if delay_decrease_temp>=delay_decrease_target:
                # possible removal
                Cost_edge_temp = utils.computeCost(Acpu_new[leaf], Amem_new[leaf], Qcpu[leaf], Qmem[leaf], Cost_cpu_edge, Cost_mem_edge)[0]
                if  Cost_edge_temp > c_max:
                    c_max = Cost_edge_temp
                    leaf_max = leaf
        if leaf_max>-1:
            logger.debug(f'cleaning microservice {leaf_max}')
            S_b_new[leaf_max] = 0
            # resource edge cloud resource shifting
            Acpu_new[leaf_max-M] = Acpu_new[leaf_max-M] + Acpu_new[leaf_max] # cloud cpu increase
            Amem_new[leaf_max-M] = Amem_new[leaf_max-M] + Amem_new[leaf_max] # cloud mem increase
            Acpu_new[leaf_max] = 0 # edge cpu decrease
            Amem_new[leaf_max] = 0 # edge mem decrease
            added_dp = added_dp - 1
        else:
            break
            
    logger.info(f"++++++++++++++++++++++++++++++")
    
    # compute final values
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

