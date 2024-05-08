# pylint: disable=C0103, C0301

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

def offload(params):

    ## INITIALIZE VARIABLES ##
    #S_edge_old (M,) vector of binary values indicating if the microservice is at the edge or not
    #Rcpu_old (2*M,) vector of CPU req by instance-set at the cloud (:M) and at the edge (M:)
    #Rmem_old (2*M,) vector of Memory req by instance-set at the cloud (:M) and at the edge (M:)
    #Fcm (M,M)microservice call frequency matrix
    #M number of microservices
    #lambd user request rate
    #Rs (M,) vector of response size of microservices
    #Di (M,) vector of internal delay of microservices
    #delay_decrease_target delay reduction target
    #RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    #Ne cloud-edge network bitrate
    #Cost_cpu_edge cost of CPU at the edge
    #Cost_mem_edge cost of Memory at the edge
    #locked (M,) vector of binary values indicating if the microservice can not change state
    #u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)

    
    S_edge_old = params['S_edge_b']
    Rcpu_old = params['Rcpu']
    Rmem_old = params['Rmem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target']
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']
    dependency_paths_b = params['dependency_paths_b']
    locked = params['locked']
    u_limit = params['u_limit']
    no_caching = params['no_caching']

    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    S_b_old[M-1] = 0  # User is not in the cloud
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    
    # SAVE CURRENT METRICS VALUES ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old,_,_,_ = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)  # Total delay of the current configuration. It includes only network delays
  
    Rcpu_edge_old_sum = np.sum(S_b_old[M:] * Rcpu_old[M:]) # Total CPU requested by instances in the edge
    Rmem_edge_old_sum = np.sum(S_b_old[M:] * Rmem_old[M:]) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_edge_old_sum # Total CPU cost at the edge
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_edge_old_sum # Total Mem cost at the edge
    Cost_edge_old = Cost_cpu_edge_old_sum + Cost_mem_edge_old_sum # Total cost at the edge

    ## BUILDING OF DEPENDENCY PATHS ##
    if dependency_paths_b is None:
        G = nx.DiGraph(Fcm) # Create microservice dependency graph 
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

        ## COMPUTE "CLOUD ONLY" DEPENDENCY PATHS ##
        for ms in range(M-1):
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            for path_n in paths_n:
                # path_n numerical id (n) of the microservices of the dependency path
                # If all microservices in the path are in the edge this path is not a cloud-only
                if all(S_b_old[M+np.array([path_n])].squeeze()==1):
                    continue
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    

    ## GREEDY ADDITION OF DEPENDECY PATHS TO EDGE CLUSTER ##
    dependency_paths_b_residual = dependency_paths_b.copy() # residual dependency path to consider in a greedy round, \Pi_r of paper
    S_b_opt = S_b_old.copy()  # S_b_opt is the best placement state computed by a greedy round
    S_b_temp = np.zeros(2*M) # S_b_temp is the temporary placement state used in a greedy round
    S_b_new = np.zeros(2*M) # S_b_new is the new placement state 
    Rcpu_opt = Rcpu_old.copy()  # Rcpu_opt is the best CPU request vector computed by a greedy round
    Rmem_opt = Rmem_old.copy()  # Rmem_opt is the best Memory request vector computed by a greedy round
    Rcpu_new = np.zeros(2*M)    # Rcpu_new is the new CPU request vector
    Rmem_new = np.zeros(2*M)    # Rmem_new is the new Memory request vector
    Rcpu_temp = np.zeros(2*M)   # Rcpu_temp is the temporary CPU request vector used in a greedy round
    Rmem_temp = np.zeros(2*M)   # Rmem_temp is the temporary Memory request vector used in a greedy round
    delay_opt = delay_old   # delay_opt is the best delay computed by a greedy round. It includes only network delays

    # result caching to accelerate computation
    delay_cache=dict()  # cache for delay computation
    Rcpu_cache=dict()   # cache for CPU request vector
    Rmem_cache=dict()   # cache for Memory request vector

    skip_delay_increase = False    # Skip delay increase states to accelerate computation wheter possible
    locking = False if locked is None else True # avoid locking control if no microservice is locked
    cost_increase_opt=0    # cost_increase_opt is the best cost increase computed by a greedy round
    delay_decrease_opt=1   # delay_decrease_opt is the best delay reduction computed by a greedy round
    
    logging.info(f"ADDING PHASE")
    while True:
        logging.info(f'-----------------------')
        w_min = float("inf") # Initialize the weight
        skip_delay_increase = False    # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)  
        np.copyto(Rcpu_new,Rcpu_opt)    # Rcpu_new is the new CPU request vector, Rcpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Rmem_new,Rmem_opt)    # Rmem_new is the new Memory request vector, Rmem_opt is the best Memory request vector computed by the previos greedy round
        delay_new = delay_opt   # delay_new is the new delay. It includes only network delays
        Cost_edge_new  = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost of the new configuration
        logging.info(f'new state {np.argwhere(S_b_new[M:]==1).squeeze()}, delay decrease {1000*(delay_old-delay_new)}, cost {Cost_edge_new}, cost increase / delay decrease {cost_increase_opt/(1000*delay_decrease_opt)}')
        
        # Check if the delay reduction is reached
        if delay_old-delay_new >= delay_decrease_target:
            #delay reduction reached
            break
        
        if len(dependency_paths_b_residual) == 0:
            # All dependency path considered no other way to reduce delay
            break

        ## GREEDY ROUND ##
        # for the next greedy round, select dependency paths providing a number of microservice upgrade not greater than u_limit
        logging.debug(f"depencency path no upgrade limit: {len(dependency_paths_b_residual)}")
        rl = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=u_limit)   # index of dependency paths with microservices upgrade less than u_limit
        logging.debug(f"depencency path with upgrade limit: {len(rl)}")
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
                logging.debug(f'cache_hit for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
                chache_hits += 1
                delay_temp = delay_cache[S_id_edge_temp]
                np.copyto(Rcpu_temp,Rcpu_cache[S_id_edge_temp])
                np.copyto(Rmem_temp,Rmem_cache[S_id_edge_temp])
                delay_decrease_temp = delay_new - delay_temp
                if skip_delay_increase and delay_decrease_temp<0:
                    continue
            else:
                Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
                Nci_temp = computeNc(Fci_temp, M, 2)    # number of instance call per user request of the temp state
                delay_temp,_,_,_ = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M) # Total delay of the temp state. It includes only network delays
                delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
                if skip_delay_increase and delay_decrease_temp<0:
                    continue
                
                # compute the cost increase adding this dependency path 
                # assumption is that cloud resource are reduce proportionally with respect to the reduction of the number of times instances are called
                np.copyto(Rcpu_temp,Rcpu_old) 
                np.copyto(Rmem_temp,Rmem_old) 
                cloud_cpu_reduction = (1-Nci_temp[:M]/Nci_old[:M]) * Rcpu_old[:M]  
                cloud_mem_reduction = (1-Nci_temp[:M]/Nci_old[:M]) * Rmem_old[:M]
                cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
                cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
                cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
                cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
                Rcpu_temp[M:] = Rcpu_temp[M:] + cloud_cpu_reduction # edge cpu increase
                Rmem_temp[M:] = Rmem_temp[M:] + cloud_mem_reduction # edge mem increase
                Rcpu_temp[:M] = Rcpu_temp[:M] - cloud_cpu_reduction # cloud cpu decrease
                Rmem_temp[:M] = Rmem_temp[:M] - cloud_mem_reduction # cloud mem decrease
                if no_caching == False:
                    delay_cache[S_id_edge_temp] = delay_temp
                    Rcpu_cache[S_id_edge_temp] = Rcpu_temp.copy() 
                    Rmem_cache[S_id_edge_temp] = Rmem_temp.copy()
                    logging.debug(f'cache insert for {np.argwhere(S_b_temp[M:]==1).squeeze()}')
            Cost_edge_temp = Cost_cpu_edge * np.sum(Rcpu_temp[M:]) + Cost_mem_edge * np.sum(Rmem_temp[M:]) # Total edge cost of the temp state
            cost_increase_temp = Cost_edge_temp - Cost_edge_new # cost increase wrt the new state
            
            # weighting
            r_delay_decrease = delay_decrease_target - (delay_old-delay_new) # residul delay to decrease wrt previous conf
            if delay_decrease_temp < 0:
                # addition provides delay increase,  weighting penalize both cost and delay increase
                w = 1e6 - cost_increase_temp * 1000 * delay_decrease_temp 
            else:
                w = cost_increase_temp /  min(1000*delay_decrease_temp, 1000*r_delay_decrease)
                skip_delay_increase = True
            
            logging.debug(f'considered state {np.argwhere(S_b_temp[M:]==1).squeeze()}, cost increase {cost_increase_temp},delay decrease {1000*delay_decrease_temp}, delay {delay_temp}, weight {w}')

            if w < w_min:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                np.copyto(Rcpu_opt,Rcpu_temp)
                np.copyto(Rmem_opt,Rmem_temp)
                cost_increase_opt = cost_increase_temp
                delay_decrease_opt = delay_decrease_temp
                delay_opt = delay_temp
                w_min = w
        
        if w_min == inf:
            # no improvement possible in the greedy round
            break

        if no_caching == False:
            logging.debug(f"cache hit prob. {chache_hits/len(dependency_paths_b_residual)}")
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
                        del Rcpu_cache[S_id_edge_temp]
                        del Rmem_cache[S_id_edge_temp]
        dependency_paths_b_residual = np.delete(dependency_paths_b_residual, PR, axis=0)
        #dependency_paths_b_residual = np.array([dependency_paths_b_residual[pr] for pr in range(len(dependency_paths_b_residual)) if pr not in PR ])
        
        # cache cleaning
        if no_caching == False:
            for S_id_edge_temp_s in list(delay_cache):
                # when a cached state do not have some edge microservice that are in new (opt) state, it will be never reused for computation
                S_id_edge_temp = id2S(int(S_id_edge_temp_s),2**M)   # binary encoded state of the edge
                if np.sum(S_b_opt[M:]) != np.sum(S_id_edge_temp * S_b_opt[M:]):
                        del delay_cache[S_id_edge_temp_s]
                        del Rcpu_cache[S_id_edge_temp_s]
                        del Rmem_cache[S_id_edge_temp_s]
                else:
                    logging.debug(f"cached state {np.argwhere(np.array(S_id_edge_temp)==1).squeeze()}")
            logging.debug(f"cache size {len(delay_cache)}")
        

    
    logging.info(f"PRUNING PHASE")
    # Remove microservice from leaves to reduce cost
    S_b_old_a = np.array(S_b_old[M:]).reshape(M,1)
    while True:
        c_max=0 # max cost of the leaf microservice to remove
        leaf_max=-1 # index of the leaf microservice to remove
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
        S_b_new_a = np.array(S_b_new[M:]).reshape(M,1)
        edge_leaves = np.logical_and(np.sum(Fci_new[M:,:], axis=1)==0, S_b_new_a==1) # edge microservice with no outgoing calls
        edge_leaves = np.logical_and(edge_leaves, S_b_old_a==0)    # old edge microservice can not be removed for incremental constraint
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        edge_leaves = edge_leaves+M # index of the edge microservice in the full state
        for leaf in edge_leaves:
            # try remove microservice
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[leaf] = 0
            Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))
            Nci_temp = computeNc(Fci_temp, M, 2)
            delay_temp,_,_,_ = computeDTot(S_b_temp, Nci_temp, Fci_temp, Di, Rs, RTT, Ne, lambd, M)
            delay_decrease_temp = delay_old - delay_temp
            if delay_decrease_temp>=delay_decrease_target:
                # possible removal
                if Rcpu_new[leaf]*Cost_cpu_edge + Rmem_new[leaf]*Cost_mem_edge > c_max:
                    leaf_max = leaf
                    c_max = Rcpu_new[leaf]*Cost_cpu_edge + Rmem_new[leaf]*Cost_mem_edge
        if leaf_max>-1:
            logging.debug(f'cleaning microservice {leaf_max}')
            S_b_new[leaf_max] = 0
            Rcpu_new[leaf_max-M] = Rcpu_new[leaf_max-M] + Rcpu_new[leaf_max] # cloud cpu increase
            Rmem_new[leaf_max-M] = Rmem_new[leaf_max-M] + Rmem_new[leaf_max] # cloud mem increase
            Rcpu_new[leaf_max] = 0 # edge cpu decrease
            Rmem_new[leaf_max] = 0 # edge mem decrease
        else:
            break
            
    logging.info(f"++++++++++++++++++++++++++++++")
    # compute final values
    n_rounds = 1
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)
    delay_decrease_new = delay_old - delay_new
    np.copyto(Rcpu_new,Rcpu_old) 
    np.copyto(Rmem_new,Rmem_old) 
    cloud_cpu_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Rcpu_old[:M]
    cloud_mem_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Rmem_old[:M]
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
    cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
    Rcpu_new[M:] = Rcpu_new[M:] + cloud_cpu_reduction # edge cpu increase
    Rmem_new[M:] = Rmem_new[M:] + cloud_mem_reduction # edge mem increase
    Rcpu_new[:M] = Rcpu_new[:M] - cloud_cpu_reduction # cloud cpu decrease
    Rmem_new[:M] = Rmem_new[:M] - cloud_mem_reduction     # cloud mem decrease
    Cost_edge_new = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
    cost_increase_new = Cost_edge_new - Cost_edge_old 

    result = dict()
    result['S_edge_b'] = S_b_new[M:].astype(int)
    result['Cost'] = Cost_edge_new
    result['delay_decrease'] = delay_decrease_new
    result['cost_increase'] = cost_increase_new
    result['n_rounds'] = n_rounds
    result['Rcpu'] = Rcpu_new
    result['Rmem'] = Rmem_new
    result['Fci'] = Fci_new
    result['Nci'] = Nci_new
    result['delay'] = delay_new
    result['di'] = di_new
    result['dn'] = dn_new
    result['rhoce'] = rhoce_new

    return result



# MAIN
parser = argparse.ArgumentParser()
parser.add_argument( '-log',
                     '--loglevel',
                     default='warning',
                     help='Provide logging level. Example --loglevel debug, default=warning' )

args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(levelname)s %(message)s')

logging.info( 'Logging now setup.' )

if __name__ == "__main__":
    # Define the input variables
    np.random.seed(150273)
    RTT = 0.0869    # RTT edge-cloud
    M = 30 # n. microservices
    delay_decrease_target = 0.03    # requested delay reduction
    lambda_val = 20     # request per second
    Ne = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one

    Cost_cpu_edge = 1 # cost of CPU at the edge
    Cost_mem_edge = 1 # cost of memory at the edge

    random=dict()
    random['n_parents'] = 3

    Fcm_range_min = 0.1 # min value of microservice call frequency 
    Fcm_range_max = 0.5 # max value of microservice call frequency 
    Rcpu_quota = 0.5    # CPU quota
    Rcpu_range_min = 1  # min value of requested CPU quota per instance-set
    Rcpu_range_max = 32 # max value of requested CPU quota per instance-set
    Rs_range_min = 1000 # min value of response size in bytes
    Rs_range_max = 50000   # max of response size in bytes
    
    Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # user has no response size
    
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
    Rcpu_void = (np.random.randint(32,size=M)+1) * Rcpu_quota
    Rcpu_void[M-1]=0   # user has no CPU request
    Rcpu_void = np.concatenate((Rcpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Rmem_void = np.zeros(2*M)
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
    
    # compute Rcpu and Rmem for the current state
    # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
    Fci = np.matrix(buildFci(S_b, Fcm, M))    # instance-set call frequency matrix of the current state
    Nci = computeNc(Fci, M, 2)    # number of instance call per user request of the current state
    Rcpu = Rcpu_void.copy()
    Rmem = Rmem_void.copy()
    cloud_cpu_decrease = (1-Nci[:M]/Nci_void[:M]) * Rcpu_void[:M]
    cloud_mem_decrease = (1-Nci[:M]/Nci_void[:M]) * Rmem_void[:M]
    cloud_cpu_decrease[np.isnan(cloud_cpu_decrease)] = 0
    cloud_mem_decrease[np.isnan(cloud_mem_decrease)] = 0
    cloud_cpu_decrease[cloud_cpu_decrease==-inf] = 0
    cloud_mem_decrease[cloud_mem_decrease==-inf] = 0
    Rcpu[M:] = Rcpu[M:] + cloud_cpu_decrease # edge cpu increase
    Rmem[M:] = Rmem[M:] + cloud_mem_decrease # edge mem increase
    Rcpu[:M] = Rcpu[:M] - cloud_cpu_decrease # cloud cpu decrease
    Rmem[:M] = Rmem[:M] - cloud_mem_decrease # cloud mem decrease
    Cost_edge = Cost_cpu_edge * np.sum(Rcpu[M:]) + Cost_mem_edge * np.sum(Rmem[M:]) # Total edge cost of the current state

    # set 0 random internal delay
    Di = np.zeros(2*M)
    
    # Call the unoffload function
    params = {
        'S_edge_b': S_edge_b,
        'Rcpu': Rcpu,
        'Rmem': Rmem,
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
        'no_caching': False
    }

        


    result = offload(params)
    print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}, rounds = {result['n_rounds']}")