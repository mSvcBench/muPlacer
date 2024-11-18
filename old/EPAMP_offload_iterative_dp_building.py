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
from S2id import S2id


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
    dependency_paths_b = params['dependency_paths_b'] if 'dependency_paths_b' in params else None
    dependency_paths_input = True if 'dependency_paths_input' in params else False
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

    ## BUILDING OF DEPENDENCY PATHS ##

    

    ## GREEDY ADDITION OF CLOUD-EDGE DEPENDECY PATHS TO EDGE CLUSTER ##
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
    Fci_new = Fci_old.copy()
    Fci_opt = Fci_old.copy()

    # result caching to accelerate computation
    delay_cache=dict()  # cache for delay computation
    rhoce_cache=dict()   # cache for rhoce computation
    expire_cache=dict() # cache for expiration round
    Acpu_cache=dict()   # cache for CPU request vector
    Amem_cache=dict()   # cache for Memory request vector
    Fci_cache=dict()    # cache for instance-set call frequency matrix

    skip_delay_increase = False    # Skip delay increase states to accelerate computation wheter possible
    locking = False if locked is None else True # avoid locking control if no microservice is locked
    
    logger.info(f"ADDING PHASE")
    round = -1

    pwd_Fcm = Fcm.copy()  
    connected=dict()
    connected[str(1)] = Fcm.copy()  
    connected[str(1)][connected[str(1)]>0]=1
    for u in range(2,u_limit+1):
        pwd_Fcm = pwd_Fcm.dot(Fcm)
        connected[str(u)] = pwd_Fcm.copy()
        connected[str(u)][connected[str(u)]>0]=1
    
    if dependency_paths_input == False:
            dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
    computed_paths=np.zeros((M,M))
    while True:
        round += 1
        logger.info(f'-----------------------')
        logger.info(f'gready round {round}')

        w_min = float("inf") # Initialize the weight
        skip_delay_increase = False    # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)  
        np.copyto(Acpu_new,Acpu_opt)    # Acpu_new is the new CPU request vector, Acpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Amem_new,Amem_opt)    # Amem_new is the new Memory request vector, Amem_opt is the best Memory request vector computed by the previos greedy round
        np.copyto(Fci_new,Fci_opt)
        delay_new = delay_opt   # delay_new is the new delay. It includes only network delays
        Cost_edge_new  = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the new configuration
        logger.info(f'new state {np.argwhere(S_b_new[M:]==1).squeeze()}, cost {Cost_edge_new}, delay decrease {1000*(delay_old-delay_new)}, cost increase {Cost_edge_new-Cost_edge_old}')
        
        # Check if the delay reduction and other constraints are reached
        
        if delay_old-delay_new >= delay_decrease_target:
            #delay reduction reached
            logger.info(f'delay reduction reached')
            break

        
        # new gready necessary
        # Compute the microservice in the edge cluster that call microservice in the cloud
        S_b_new_a = np.array(S_b_new[M:]).reshape(M,1)
        edge_gates = np.logical_and(np.sum(Fci_new[M:,:M], axis=1)>0, S_b_new_a==1) # edge microservice with outgoing calls to cloud microservices
        edge_gates = np.argwhere(edge_gates)[:,0]
        # compute partial dependency paths from the gates at max distance u_limit hops
        ## building of partial dependency paths
        if dependency_paths_input == False:
            for u in range(1,u_limit+1):
                for e in edge_gates:
                    for ms in np.argwhere(connected[str(u)][e]==1):
                        if computed_paths[e,ms] != 1:
                            computed_paths[e,ms] = 1
                            if S_b_new[ms+M] == 0:
                                S_b_temp = S_b_new.copy()
                                S_b_temp[ms+M] = 1
                                paths_n = list(nx.all_simple_paths(G, source=e, target=ms))
                                for path_n in paths_n:
                                    path_b = np.zeros((1,M),int)
                                    path_b[0,path_n] = 1
                                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)

        logger.info(f'considered dependency paths {len(dependency_paths_b)}')
        if len(dependency_paths_b) == 0:
            # All dependency path considered no other way to reduce delay
            logger.info(f'All dependency path considered no other way to reduce delay')
            break

        ## GREEDY ROUND ##
        # ### expanding ring
        # for diameter in range(2,10):
        #     # u_limit complexity reduction
        #     # for the next greedy round, select dependency paths providing a number of microservice upgrade not greater than u_limit
        #     logger.debug(f"number of depencency paths without upgrade limit: {len(dependency_paths_b_residual)}")
        #     rl = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=diameter)   # index of dependency paths with microservices upgrade less than u_limit
        #     logger.debug(f"number of depencency paths with upgrade limit: {len(rl)}")
        #     ### random starting solution that decrease delay
        #     np.random.shuffle(rl)
        #     addedd_dp = list()
        #     for j in np.random.shuffle(rl):
        #         addedd_dp.append(j)
        #         path_b = dependency_paths_b_residual[j]
        #         path_n = np.argwhere(path_b.flatten()==1).squeeze()
        #         np.copyto(S_b_temp, S_b_new)
        #         S_b_new[M+path_n] = 1
        #         S_id_edge_temp=str(S2id(S_b_new[M:])) # id for the cache entry
        #         if locking:
        #             if not np.equal(S_b_new[M:]*locked, S_b_old[M:]*locked).all(): # if a locked microservice is moved, skip
        #                 continue
        #         if S_id_edge_temp in delay_cache:
        #             logger.debug(f'cache_hit for {np.argwhere(S_b_new[M:]==1).squeeze()}')
        #             cache_hit += 1
        #             if expire_cache[S_id_edge_temp] == round:
        #                 # state already considered in the round
        #                 continue
        #             delay_temp = delay_cache[S_id_edge_temp]
        #             Acpu_temp = np.copy(Acpu_cache[S_id_edge_temp])
        #             Amem_temp = np.copy(Amem_cache[S_id_edge_temp])
        #             rhoce = rhoce_cache[S_id_edge_temp]
        #             expire_cache[S_id_edge_temp] = round
        #         else:
        #             Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
        #             Nci_temp = computeNc(Fci_temp, M, 2)    # number of instance call per user request of the temp state
        #             delay_temp,_,_,rhoce = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0)) # Total delay of the temp state. It includes only network delays
        #             # compute the cost increase adding this dependency path 
        #             # assumption is that cloud resource are reduce proportionally with respect to the reduction of the number of times instances are called
        #             utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_old,Amem_old,Nci_old)
        #             # caching
        #             delay_cache[S_id_edge_temp] = delay_temp
        #             rhoce_cache[S_id_edge_temp] = rhoce
        #             Acpu_cache[S_id_edge_temp]=np.copy(Acpu_temp)
        #             Amem_cache[S_id_edge_temp]=np.copy(Amem_temp)
        #             expire_cache[S_id_edge_temp] = round
        #             logger.debug(f'cache insert for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
        #         if delay_temp < delay_new:
        #             # update best state of the expanding ring phase
        #             np.copyto(S_b_new,S_b_temp)
        #             delay_new = delay_temp
        #             break
            
            
            
        cache_hit = 0    
        for path_b in dependency_paths_b :
            # merging path_b and S_b_new into S_b_temp
            path_n = np.argwhere(path_b.flatten()==1).squeeze() # numerical id of the microservices of the dependency path
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[M+path_n] = 1
            S_id_edge_temp=str(S2id(S_b_temp[M:])) # id for the cache entry
            
            #check looked microservices
            if locking:
                if not np.equal(S_b_temp[M:]*locked, S_b_old[M:]*locked).all(): # if a locked microservice is moved, skip
                    continue
            
            if S_id_edge_temp in delay_cache:
                logger.debug(f'cache_hit for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
                cache_hit += 1
                if expire_cache[S_id_edge_temp] == round:
                    # state already considered in the round
                    continue
                delay_temp = delay_cache[S_id_edge_temp].copy()
                Acpu_temp = Acpu_cache[S_id_edge_temp].copy()
                Amem_temp = Amem_cache[S_id_edge_temp].copy()
                Fci_temp = Fci_cache[S_id_edge_temp].copy()
                rhoce = rhoce_cache[S_id_edge_temp].copy()
                delay_decrease_temp = delay_new - delay_temp
                expire_cache[S_id_edge_temp] = round
                if skip_delay_increase and delay_decrease_temp<0:
                    logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()} skipped for negative delay decrease')
                    continue
            else:
                Fci_temp = np.matrix(buildFi(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
                Nci_temp = computeN(Fci_temp, M, 2)    # number of instance call per user request of the temp state
                delay_temp,_,_,rhoce = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0)) # Total delay of the temp state. It includes only network delays
                delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
                if skip_delay_increase and delay_decrease_temp<0:
                    logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()} skipped for negative delay decrease')
                    continue
                
                # compute the cost increase adding this dependency path 
                # assumption is that cloud resource are reduce proportionally with respect to the reduction of the number of times instances are called
                utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_old,Amem_old,Nci_old)
                Cost_edge_temp = utils.computeCost(Acpu_temp[M:], Amem_temp[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the temp state
                cost_increase_temp = Cost_edge_temp - Cost_edge_new # cost increase wrt the new state
                
                # caching
                delay_cache[S_id_edge_temp] = delay_temp.copy()
                rhoce_cache[S_id_edge_temp] = rhoce.copy()
                Acpu_cache[S_id_edge_temp]=Acpu_temp.copy()
                Amem_cache[S_id_edge_temp]=Amem_temp.copy()
                Fci_cache[S_id_edge_temp]=Fci_temp.copy()
                expire_cache[S_id_edge_temp] = round
                logger.debug(f'cache insert for {np.argwhere(S_b_temp[M:]==1).squeeze()}')

            # weighting
            r_delay_decrease = delay_decrease_target - (delay_old-delay_new) # residul delay to decrease wrt previous conf
            if rhoce == 1 or delay_decrease_temp <= 0:
                # addition provides delay increase,  weighting penalize both cost and delay increase
                # dp_centrality = np.sum(np.sum(dependency_paths_b,axis=1)[added_ms]) # centrality of the added microservices in the dependency graph
                # w = 1e6 + cost_increase_temp/dp_centrality
                # w = 1e6 - cost_increase_temp * 1000 * delay_decrease_temp   # 1000 used to move delay in the ms scale
                w = 1e6 - cost_increase_temp * 1000 * delay_decrease_temp   # 1000 used to move delay in the ms scale
                # w = 1e6 - sum(Nci_temp[M:]-Nci_old[M:]) # prefer mostrly used microservices 
            else:
                w = cost_increase_temp /  max(min(1000*delay_decrease_temp, 1000*r_delay_decrease),1e-3) # 1e-3 used to avoid division by zero
                skip_delay_increase = True
            
            logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()}, cost increase {cost_increase_temp},delay decrease {1000*delay_decrease_temp}, delay {delay_temp}, weight {w}')

            if w < w_min:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                Acpu_opt = np.copy(Acpu_temp)
                Amem_opt = np.copy(Amem_temp)
                Fci_opt = np.copy(Fci_temp)
                delay_opt = delay_temp
                w_min = w
        logger.info(f'chache hit probability {cache_hit/len(dependency_paths_b)}')
        if w_min == inf:
            # no improvement possible in the greedy round
            logger.info(f'no improvement possible in the greedy round')
            break

        # Prune not considered dependency paths whose microservices are going to be contained in the edge to accelerate computation
        PR = []
        duplicateID = list()
        for pr,path_b in enumerate(dependency_paths_b):
            if np.sum(path_b) == np.sum(path_b * S_b_opt[M:]):
                # dependency path already fully included at edge
                logger.debug(f'pruning dependency path {np.argwhere(path_b>0).flatten()} already fully included at edge')
                PR.append(pr)
            else:
                path_n = np.argwhere(path_b.flatten()==1).squeeze()
                S_b_temp = S_b_opt.copy() 
                S_b_temp[path_n+M]=1
                S_id_edge_temp=str(S2id(S_b_temp[M:]))
                if S_id_edge_temp in duplicateID:
                    PR.append(pr)
                else:
                    duplicateID.append(S_id_edge_temp)
        dependency_paths_b = np.delete(dependency_paths_b, PR, axis=0)
        # cache cleaning
        for key in list(delay_cache.keys()):
            if expire_cache[key] + 10 < round:
                del delay_cache[key]
                del rhoce_cache[key]
                del Acpu_cache[key]
                del Amem_cache[key]
                del expire_cache[key]
    logger.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_old_a = np.array(S_b_old[M:]).reshape(M,1)
    while True:
        w_opt = -1 # weight of the best removal
        leaf_best = -1 # index of the leaf microservice to remove
        S_b_temp = np.zeros(2*M)
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
        Nci_new = computeN(Fci_new, M, 2)
        S_b_new_a = np.array(S_b_new[M:]).reshape(M,1)
        delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
        utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
        Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
        edge_leaves = np.logical_and(np.sum(Fci_new[M:,M:], axis=1)==0, S_b_new_a==1) # edge microservice with no outgoing calls
        if (no_evolutionary==False):
            edge_leaves = np.logical_and(edge_leaves, S_b_old_a==0)    # old edge microservice can not be removed for incremental constraint
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        edge_leaves = edge_leaves+M # index of the edge microservice in the full state
        logger.info(f'pruning candidates {edge_leaves-M}')
        for leaf in edge_leaves:
            # try remove microservice
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[leaf] = 0
            Acpu_temp = np.zeros(2*M)
            Amem_temp = np.zeros(2*M)
            Fci_temp = np.matrix(buildFi(S_b_temp, Fcm, M))
            Nci_temp = computeN(Fci_temp, M, 2)
            delay_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))[0]
            delay_increase_temp = delay_temp - delay_new
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            Cost_edge_temp = utils.computeCost(Acpu_temp[M:], Amem_temp[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
            cost_decrease = Cost_edge_new - Cost_edge_temp
            w = cost_decrease/delay_increase_temp
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            
            if w>w_opt and delay_old - delay_temp > delay_decrease_target:
                # possible removal
                w_opt = w
                leaf_best = leaf
                delay_reduction = delay_old - delay_temp
        if leaf_best>-1:
            logger.info(f'pruned microservice {leaf_best-M}, delay reduction: {delay_reduction}')
            S_b_new[leaf_best] = 0
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

