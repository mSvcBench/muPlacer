# pylint: disable=C0103, C0301
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import networkx as nx
from computeNc import computeNc
from buildFci import buildFci
from S2id import S2id
from id2S import id2S
from numpy import inf
from computeDTot import computeDTot
import logging
import sys
import argparse


np.seterr(divide='ignore', invalid='ignore')

# Set up logger
logger = logging.getLogger('EPAMP_offload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP offload %(levelname)s %(message)s'))
logger.propagate = False

def offload(params):

    ## INITIALIZE VARIABLES ##
    #S_edge_old (M,) vector of binary values indicating if the microservice is at the edge or not
    #Acpu_old (2*M,) vector of CPU req by instance-set at the cloud (:M) and at the edge (M:)
    #Amem_old (2*M,) vector of Memory req by instance-set at the cloud (:M) and at the edge (M:)
    #Fcm (M,M)microservice call frequency matrix
    #M number of microservices
    #lambd user request rate
    #Rs (M,) vector of response size of microservices
    #Rsd (M,) vector of average delay of response sizes Rs
    #Di (M,) vector of internal delay of microservices
    #delay_decrease_target delay reduction target
    #RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    #Ne cloud-edge network bitrate
    #Cost_cpu_edge cost of CPU at the edge
    #Cost_mem_edge cost of Memory at the edge
    #locked (M,) vector of binary values indicating if the microservice can not change state
    #u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)
    # no_caching if True, disable caching of the delay computation
    # no_evolutionary if True, disable the removal of microservices from the edge to reduce cost
    # max_added_dp max added dependency path before stopping the greedy iteration
    # min_added_dp min missing dependency path to add before stopping the greedy iteration
    # Qmem (M,) memory quantum in bytes
    # Qcpu (M,) CPU quantum in cpu sec
    def numpy_array_to_list(numpy_array):
        return list(numpy_array.flatten())
    
    def qz(x,y):
        res = np.zeros(len(x))
        z = np.argwhere(y==0)
        res[z] = x[z]
        nz = np.argwhere(y>0)
        res[nz] = np.ceil(x[nz]/y[nz])*y[nz]
        return res

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
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(M) # memory quantum in bytes
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(M) # CPU quantum in cpu sec
    max_added_dp = params['max_added_dp'] if 'max_added_dp' in params else 1000000 # maximum number of dependency path added in the greedy iteration
    min_added_dp = params['min_added_dp'] if 'min_added_dp' in params else 0 # minimum number of dependency path added in the greeedy iteration, negative value means that the minimum is equal to the whole set of dependency path minus the passed value
    
    # optional paramenters
    Di = params['Di'] if 'Di' in params else np.zeros(M)
    dependency_paths_b = params['dependency_paths_b'] if 'dependency_paths_b' in params else None
    locked = params['locked'] if 'locked' in params else None
    u_limit = params['u_limit'] if 'u_limit' in params else M
    no_caching = params['no_caching'] if 'no_caching' in params else False
    no_evolutionary = params['no_evolutionary'] if 'no_evolutionary' in params else False
    Rsd = params['Rsd'] if 'Rsd' in params else np.empty(0)

    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    S_b_old[M-1] = 0  # User is not in the cloud
    
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
    
    # SAVE CURRENT METRICS VALUES ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old,_,_,_ = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M, Rsd)  # Total delay of the current configuration. It includes only network delays
  
    Acpu_edge_old_sum = np.sum(qz(S_b_old[M:] * Acpu_old[M:],Qcpu[M:])) # Total CPU requested by instances in the edge
    Amem_edge_old_sum = np.sum(qz(S_b_old[M:] * Amem_old[M:],Qmem[M:])) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Acpu_edge_old_sum # Total CPU cost at the edge
    Cost_mem_edge_old_sum = Cost_mem_edge * Amem_edge_old_sum # Total Mem cost at the edge
    Cost_edge_old = Cost_cpu_edge_old_sum + Cost_mem_edge_old_sum # Total cost at the edge

    ## BUILDING OF DEPENDENCY PATHS ##
    if dependency_paths_b is None:
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

        ## COMPUTE "CLOUD JOINED" DEPENDENCY PATHS ##
        for ms in range(M-1):
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            for path_n in paths_n:
                # path_n numerical id (n) of the microservices of the dependency path
                # If all microservices in the path are in the edge this path is not a cloud-joined path
                if all(S_b_old[M+np.array([path_n])].squeeze()==1):
                    continue
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    

    ## GREEDY ADDITION OF CLOUD JOINED DEPENDECY PATHS TO EDGE CLUSTER ##
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

    # result caching to accelerate computation
    delay_cache=dict()  # cache for delay computation
    Acpu_cache=dict()   # cache for CPU request vector
    Amem_cache=dict()   # cache for Memory request vector

    skip_delay_increase = False    # Skip delay increase states to accelerate computation wheter possible
    locking = False if locked is None else True # avoid locking control if no microservice is locked
    cost_increase_opt=0    # cost_increase_opt is the best cost increase computed by a greedy round
    delay_decrease_opt=1   # delay_decrease_opt is the best delay reduction computed by a greedy round
    logger.info(f"ADDING PHASE")

    if min_added_dp < 0:
        # min_added_dp is the number of dependency path to add before stopping the greedy iteration
        # negative value means that the minimum is equal to the whole set of dependency path minus the passed value
        min_added_dp = len(dependency_paths_b_residual) + min_added_dp
    while True:
        logger.info(f'-----------------------')
        w_min = float("inf") # Initialize the weight
        skip_delay_increase = False    # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)  
        np.copyto(Acpu_new,Acpu_opt)    # Acpu_new is the new CPU request vector, Acpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Amem_new,Amem_opt)    # Amem_new is the new Memory request vector, Amem_opt is the best Memory request vector computed by the previos greedy round
        delay_new = delay_opt   # delay_new is the new delay. It includes only network delays
        Cost_edge_new  = Cost_cpu_edge * np.sum(qz(Acpu_new[M:],Qcpu[M:])) + Cost_mem_edge * np.sum(qz(Amem_new[M:],Qmem[M:])) # Total edge cost of the new configuration
        logger.info(f'new state {np.argwhere(S_b_new[M:]==1).squeeze()}, delay decrease {1000*(delay_old-delay_new)}, cost {Cost_edge_new}, cost increase / delay decrease {cost_increase_opt/(1000*delay_decrease_opt)}')
        
        # Check if the delay reduction and other constraints are reached
        added_dp = len(dependency_paths_b)-len(dependency_paths_b_residual)

        if delay_old-delay_new >= delay_decrease_target and added_dp >= min_added_dp:
            #delay reduction reached with minimum number of dependency paths added
            break

        if added_dp >= max_added_dp:
            # max number of dependency paths to add reached
            break

        if len(dependency_paths_b_residual) == 0:
            # All dependency path considered no other way to reduce delay
            break

        ## GREEDY ROUND ##
        # for the next greedy round, select dependency paths providing a number of microservice upgrade not greater than u_limit
        logger.debug(f"depencency path no upgrade limit: {len(dependency_paths_b_residual)}")
        rl = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=u_limit)   # index of dependency paths with microservices upgrade less than u_limit
        logger.debug(f"depencency path with upgrade limit: {len(rl)}")
        chache_hits = 0 # cache hit counter
        for path_b in dependency_paths_b_residual[rl] :
            # merging path_b and S_b_new into S_b_temp
            path_n = np.argwhere(path_b.flatten()==1).squeeze() 
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[M+path_n] = 1
            
            #check looked microservices
            if locking:
                if not np.equal(S_b_temp[M:]*locked, S_b_old[M:]*locked).all(): # if a locked microservice is moved, skip
                    continue
            if no_caching == False:
                S_id_edge_temp=str(S2id(S_b_temp[M:]))  # decimal encoded id of the edge state
            if no_caching==False and S_id_edge_temp in delay_cache:
                logger.debug(f'cache_hit for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
                chache_hits += 1
                delay_temp = delay_cache[S_id_edge_temp]
                np.copyto(Acpu_temp,Acpu_cache[S_id_edge_temp])
                np.copyto(Amem_temp,Amem_cache[S_id_edge_temp])
                delay_decrease_temp = delay_new - delay_temp
                if skip_delay_increase and delay_decrease_temp<0:
                    continue
            else:
                Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
                Nci_temp = computeNc(Fci_temp, M, 2)    # number of instance call per user request of the temp state
                delay_temp,_,_,_ = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, Rsd) # Total delay of the temp state. It includes only network delays
                delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
                if skip_delay_increase and delay_decrease_temp<0:
                    continue
                
                # compute the cost increase adding this dependency path 
                # assumption is that cloud resource are reduce proportionally with respect to the reduction of the number of times instances are called
                np.copyto(Acpu_temp,Acpu_old) 
                np.copyto(Amem_temp,Amem_old) 
                cloud_cpu_reduction = (1-Nci_temp[:M]/Nci_old[:M]) * Acpu_old[:M]  
                cloud_mem_reduction = (1-Nci_temp[:M]/Nci_old[:M]) * Amem_old[:M]
                cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
                cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
                cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
                cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
                Acpu_temp[M:] = Acpu_temp[M:] + cloud_cpu_reduction # edge cpu increase
                Amem_temp[M:] = Amem_temp[M:] + cloud_mem_reduction # edge mem increase
                Acpu_temp[:M] = Acpu_temp[:M] - cloud_cpu_reduction # cloud cpu decrease
                Amem_temp[:M] = Amem_temp[:M] - cloud_mem_reduction # cloud mem decrease
                if no_caching == False:
                    delay_cache[S_id_edge_temp] = delay_temp
                    Acpu_cache[S_id_edge_temp] = Acpu_temp.copy() 
                    Amem_cache[S_id_edge_temp] = Amem_temp.copy()
                    logger.debug(f'cache insert for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
            Cost_edge_temp = Cost_cpu_edge * np.sum(qz(Acpu_temp[M:],Qcpu[M:])) + Cost_mem_edge * np.sum(qz(Amem_temp[M:],Qmem[M:])) # Total edge cost of the temp state
            cost_increase_temp = Cost_edge_temp - Cost_edge_new # cost increase wrt the new state
            
            # weighting
            r_delay_decrease = delay_decrease_target - (delay_old-delay_new) # residul delay to decrease wrt previous conf
            if delay_decrease_temp < 0:
                # addition provides delay increase,  weighting penalize both cost and delay increase
                w = 1e6 - cost_increase_temp * 1000 * delay_decrease_temp 
            else:
                w = cost_increase_temp /  max(min(1000*delay_decrease_temp, 1000*r_delay_decrease),1e-3) # 1e-3 used to avoid division by zero
                skip_delay_increase = True
            
            logger.debug(f'considered state {np.argwhere(S_b_temp[M:]==1).squeeze()}, cost increase {cost_increase_temp},delay decrease {1000*delay_decrease_temp}, delay {delay_temp}, weight {w}')

            if w < w_min:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                np.copyto(Acpu_opt,Acpu_temp)
                np.copyto(Amem_opt,Amem_temp)
                cost_increase_opt = cost_increase_temp
                delay_decrease_opt = delay_decrease_temp
                delay_opt = delay_temp
                w_min = w
        
        if w_min == inf:
            # no improvement possible in the greedy round
            break

        if no_caching == False:
            logger.debug(f"cache hit prob. {chache_hits/len(dependency_paths_b_residual)}")
        # Prune not considered dependency paths whose microservices are going to be contained in the edge to accelerate computation
        PR = []
        for pr,path_b in enumerate(dependency_paths_b_residual):
            if np.sum(path_b) == np.sum(path_b * S_b_opt[M:]):
                # dependency path already fully included at edge
                PR.append(pr)
                # cache cleaning
                if no_caching == False:
                    S_id_edge_temp = str(S2id(S_b_temp[M:]))
                    if S_id_edge_temp in delay_cache:
                        del delay_cache[S_id_edge_temp]
                        del Acpu_cache[S_id_edge_temp]
                        del Amem_cache[S_id_edge_temp]
        dependency_paths_b_residual = np.delete(dependency_paths_b_residual, PR, axis=0)
        #dependency_paths_b_residual = np.array([dependency_paths_b_residual[pr] for pr in range(len(dependency_paths_b_residual)) if pr not in PR ])
        
        # cache cleaning
        if no_caching == False:
            for S_id_edge_temp_s in list(delay_cache):
                # when a cached state do not have some edge microservice that are in new (opt) state, it will be never reused for computation
                S_id_edge_temp = id2S(int(S_id_edge_temp_s),2**M)   # binary encoded state of the edge
                if np.sum(S_b_opt[M:]) != np.sum(S_id_edge_temp * S_b_opt[M:]):
                        del delay_cache[S_id_edge_temp_s]
                        del Acpu_cache[S_id_edge_temp_s]
                        del Amem_cache[S_id_edge_temp_s]
                else:
                    logger.debug(f"cached state {np.argwhere(np.array(S_id_edge_temp)==1).squeeze()}")
            logger.debug(f"cache size {len(delay_cache)}")
        

    
    logger.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_old_a = np.array(S_b_old[M:]).reshape(M,1)
    while True:
        c_max=0 # max cost of the leaf microservice to remove
        leaf_max=-1 # index of the leaf microservice to remove
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
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
            Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))
            Nci_temp = computeNc(Fci_temp, M, 2)
            delay_temp,_,_,_ = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M, Rsd)
            delay_decrease_temp = delay_old - delay_temp
            if delay_decrease_temp>=delay_decrease_target:
                # possible removal
                if qz(Acpu_new[leaf],Qcpu[leaf-M])*Cost_cpu_edge + qz(Amem_new[leaf],Qmem[leaf-M])*Cost_mem_edge > c_max:
                    leaf_max = leaf
                    c_max = qz(Acpu_new[leaf],Qcpu[leaf-M])*Cost_cpu_edge + qz(Amem_new[leaf],Qmem[leaf-M])*Cost_mem_edge
        if leaf_max>-1:
            logger.debug(f'cleaning microservice {leaf_max}')
            S_b_new[leaf_max] = 0
            Acpu_new[leaf_max-M] = Acpu_new[leaf_max-M] + Acpu_new[leaf_max] # cloud cpu increase
            Amem_new[leaf_max-M] = Amem_new[leaf_max-M] + Amem_new[leaf_max] # cloud mem increase
            Acpu_new[leaf_max] = 0 # edge cpu decrease
            Amem_new[leaf_max] = 0 # edge mem decrease
        else:
            break
            
    logger.info(f"++++++++++++++++++++++++++++++")
    # compute final values
    n_rounds = 1
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M, Rsd)
    delay_decrease_new = delay_old - delay_new
    np.copyto(Acpu_new,Acpu_old) 
    np.copyto(Amem_new,Amem_old) 
    cloud_cpu_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Acpu_old[:M]
    cloud_mem_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Amem_old[:M]
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
    cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
    Acpu_new[M:] = Acpu_new[M:] + cloud_cpu_reduction # edge cpu increase
    Amem_new[M:] = Amem_new[M:] + cloud_mem_reduction # edge mem increase
    Acpu_new[:M] = Acpu_new[:M] - cloud_cpu_reduction # cloud cpu decrease
    Amem_new[:M] = Amem_new[:M] - cloud_mem_reduction     # cloud mem decrease
    Cost_edge_new = Cost_cpu_edge * np.sum(qz(Acpu_new[M:],Qcpu[M:])) + Cost_mem_edge * np.sum(qz(Amem_new[M:],Qmem[M:])) # Total edge cost
    cost_increase_new = Cost_edge_new - Cost_edge_old

    result_edge = dict()
    
    # extra information
    result_edge['S_edge_b'] = S_b_new[M:].astype(int)
    result_edge['Cost'] = Cost_edge_new
    result_edge['delay_decrease'] = delay_decrease_new
    result_edge['cost_increase'] = cost_increase_new
    result_edge['n_rounds'] = n_rounds
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
    result_cloud['placement'] = numpy_array_to_list(np.argwhere(S_b_new[:M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"


    result_edge['to-apply'] = numpy_array_to_list(np.argwhere(S_b_new[M:]-S_b_old[M:]>0))
    result_edge['to-delete'] = numpy_array_to_list(np.argwhere(S_b_old[M:]-S_b_new[M:]>0))
    result_edge['placement'] = numpy_array_to_list(np.argwhere(S_b_new[M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_edge['delay_decrease'] < delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    return result_return



# MAIN

def main():
    # small simulation to test the offload function
    def qz(x,y):
        res = np.zeros(len(x))
        z = np.argwhere(y==0)
        res[z] = x[z]
        nz = np.argwhere(y>0)
        res[nz] = np.ceil(x[nz]/y[nz])*y[nz]
        return res
    # Define the input variables
    np.random.seed(150271)
    RTT = 0.106    # RTT edge-cloud
    M = 30 # n. microservices
    delay_decrease_target = 0.08    # requested delay reduction
    lambda_val = 50     # request per second
    Ne = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one

    Cost_cpu_edge = 1 # cost of CPU at the edge
    Cost_mem_edge = 1 # cost of memory at the edge

    random=dict()
    random['n_parents'] = 3

    Fcm_range_min = 0.1 # min value of microservice call frequency 
    Fcm_range_max = 0.5 # max value of microservice call frequency 
    Acpu_quota = 0.5    # CPU quota
    Acpu_range_min = 1  # min value of requested CPU quota per instance-set
    Acpu_range_max = 32 # max value of requested CPU quota per instance-set
    Rs_range_min = 1000 # min value of response size in bytes
    Rs_range_max = 50000   # max of response size in bytes
    
    Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # user has no response size
    Rsd = None
    
    # build dependency graph
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    for i in range(1,M-1):
        n_parent=np.random.randint(1,random['n_parents'])
        for j in range(n_parent):
            a = np.random.randint(i)
            Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(0.1,0.5) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)
    
    # add x dependency path at random
    G = nx.DiGraph(Fcm) # Create microservice dependency graph 
    dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
    for ms in range(M-1):
        paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        for path_n in paths_n:
            path_b = np.zeros((1,M),int)
            path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
            dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    l = len(dependency_paths_b)
    x = 10
    random_values = np.random.choice(range(l), size=x, replace=False)
    for j in random_values:
        S_edge_b = np.minimum(S_edge_b + dependency_paths_b[j],1)
    S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
    S_b[M-1] = 0  # User is not in the cloud
    # set random values for CPU and memory requests
    Acpu_void = (np.random.randint(32,size=M)+1) * Acpu_quota
    Acpu_void[M-1]=0   # user has no CPU request
    Acpu_void = np.concatenate((Acpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Amem_void = np.zeros(2*M)
    Qcpu = np.ones(M)   # CPU quantum in cpu sec
    Qmem = np.zeros(M)   # Memory quantum in bytes
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
    
    # compute Acpu and Amem for the current state
    # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
    Fci = np.matrix(buildFci(S_b, Fcm, M))    # instance-set call frequency matrix of the current state
    Nci = computeNc(Fci, M, 2)    # number of instance call per user request of the current state
    Acpu = Acpu_void.copy()
    Amem = Amem_void.copy()
    cloud_cpu_decrease = (1-Nci[:M]/Nci_void[:M]) * Acpu_void[:M]
    cloud_mem_decrease = (1-Nci[:M]/Nci_void[:M]) * Amem_void[:M]
    cloud_cpu_decrease[np.isnan(cloud_cpu_decrease)] = 0
    cloud_mem_decrease[np.isnan(cloud_mem_decrease)] = 0
    cloud_cpu_decrease[cloud_cpu_decrease==-inf] = 0
    cloud_mem_decrease[cloud_mem_decrease==-inf] = 0
    Acpu[M:] = Acpu[M:] + cloud_cpu_decrease # edge cpu increase
    Amem[M:] = Amem[M:] + cloud_mem_decrease # edge mem increase
    Acpu[:M] = Acpu[:M] - cloud_cpu_decrease # cloud cpu decrease
    Amem[:M] = Amem[:M] - cloud_mem_decrease # cloud mem decrease
    Cost_edge = Cost_cpu_edge * np.sum(qz(Acpu[M:],Qcpu[M:])) + Cost_mem_edge * np.sum(qz(Amem[M:],Qmem[M:])) # Total edge cost of the current state

    # set 0 random internal delay
    Di = np.zeros(2*M)
    
    # Call the offload function
    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Acpu,
        'Amem': Amem,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 2,
        'no_caching': False,
        'Qcpu': np.ones(M),
        'Qmem': np.ones(M),
        'no_evolutionary': False,
        'max_added_dp': 1
    }

        
    result_list = offload(params)
    result=result_list[1]
    print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}, rounds = {result['n_rounds']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '-log',
                     '--loglevel',
                     default='warning',
                     help='Provide logging level. Example --loglevel debug, default=warning' )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(asctime)s EPAMP offload %(levelname)s %(message)s')

    logging.info( 'Logging now setup.' )
    main()

