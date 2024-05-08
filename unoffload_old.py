# pylint: disable=C0103, C0301

import datetime
import numpy as np
import networkx as nx
from computeNc import computeNc
from buildFci import buildFci
from S2id import S2id
from id2S import id2S
from numpy import inf
from computeDTot import computeDnTot

np.seterr(divide='ignore', invalid='ignore')

def unoffload(Rcpu_old, Rmem_old, Fcm, M, lambd, Rs, S_edge_old, delay_increase_target, RTT, Ne, Cost_cpu_edge, Cost_mem_edge, u_limit):

    ## INITIALIZE VARIABLES ##
    #Rcpu_old (2*M,) vector of CPU req by instance-set at the cloud (:M) and at the edge (M:)
    #Rmem_old (2*M,) vector of Memory req by instance-set at the cloud (:M) and at the edge (M:)
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
    #u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)

     
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    S_b_old[M-1] = 0  # User is not in the cloud
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    
    # SAVE CURRENT METRICS VALUES ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old = computeDnTot(S_b_old, Nci_old, Fci_old, Rs, RTT, Ne, lambd, M)  # Total delay of the current configuration. It includes only network delays
  
    Rcpu_edge_old_sum = np.sum(S_b_old[M:] * Rcpu_old[M:]) # Total CPU requested by instances in the edge
    Rmem_edge_old_sum = np.sum(S_b_old[M:] * Rmem_old[M:]) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_edge_old_sum # Total CPU cost at the edge
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_edge_old_sum # Total Mem cost at the edge
    Cost_edge_old = Cost_cpu_edge_old_sum + Cost_mem_edge_old_sum # Total cost at the edge

    ## BUILDING OF DEPENDENCY PATHS ##
    G = nx.DiGraph(Fcm) # Create microservice dependency graph 
    dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

    ## COMPUTE "EDGE ONLY" DEPENDENCY PATHS ##
    for ms in range(M-1):
        paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        for path_n in paths_n:
            # path_n numerical id (n) of the microservices of the dependency path
            # If not all microservices in the path are in the edge, this path is not a edge-only
            if not all(S_b_old[M+np.array([path_n])].squeeze()==1):
                continue
            else:
                path_b = np.zeros((1,M),int)
                path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    

    ## GREEDY ADDITION OF DEPENDECY PATHS TO EDGE CLUSTER ##
    dependency_paths_b_residual = dependency_paths_b.copy() # residual dependency path to consider in a greedy round, \Pi_r of paper
    S_b_opt = S_b_old  # S_b_opt is the best placement state computed by a greedy round
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

    debug = False
    debug2 = False
    debug_cache = False
    skip_delay_increase = False    # Skip delay increase to accelerate computation when possible
    cost_decrease_opt=0    # cost_decrease_opt is the best cost decrease computed by a greedy round
    delay_increase_opt=0   # delay_increase_opt is the best delay increase computed by a greedy round
    
    print(f"PRUNING PHASE") if debug_cache else 0
    while True:
        print(f'-----------------------') if debug else 0
        w_max = -inf # Initialize the weight
        skip_delay_increase = False    # Skip delay increase to accelerate computation if a solution with delay decrease exists
        np.copyto(S_b_new,S_b_opt)  
        np.copyto(Rcpu_new,Rcpu_opt)    # Rcpu_new is the new CPU request vector, Rcpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Rmem_new,Rmem_opt)    # Rmem_new is the new Memory request vector, Rmem_opt is the best Memory request vector computed by the previos greedy round
        
        # if the delay increase has been exceeded holds previous configuration
        if delay_opt - delay_old >= delay_increase_target:
            break
        
        delay_new = delay_opt   # delay_new is the new delay. It includes only network delays
        Cost_edge_new  = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost of the new configuration
        print(f'new ms {np.argwhere(S_b_new[M:]==1).squeeze()}, delta_delay {1000*(delay_old-delay_new)}, cost {Cost_edge_new}, delta_cost/delta_delay {cost_decrease_opt/(1000*delay_increase_opt)}') if debug else 0
    
        if len(dependency_paths_b_residual) == 0:
            # All dependency paths considered no other way to increase delay
            break

        ## GREEDY ROUND ##
        # for the next greedy round, select dependency paths providing a number of microservice upgrade not greater than u_limit
        # print(f"depencency path no depth: {len(dependency_paths_b_residual)}") if debug else 0
        # r = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=u_limit).squeeze()
        # print(f"depencency path with depth: {len(r)}") if debug else 0
        chache_hits = 0 # cache hit counter
        r = range(len(dependency_paths_b_residual))
        for i in r :
            # removing path_b from S_b_new into S_b_temp
            np.copyto(S_b_temp,S_b_new)
            dependency_paths_b_join = [dependency_paths_b_residual[idx] for idx in r if idx != i]
            S_b_temp[M:] = np.minimum(np.sum(dependency_paths_b_join,axis=0),1)
            print(np.argwhere(S_b_temp[M:]==1).squeeze())
            print(np.argwhere(dependency_paths_b_residual[i]==1).squeeze())
            if np.equal(S_b_temp[M:],S_b_new[M:]).any():
                # dependency path can not be fully removed as the new state is the same
                continue
            S_id_edge_temp=str(S2id(S_b_temp[M:]))  # decimal encoded id of the edge state 
            if S_id_edge_temp in delay_cache:
                print(f'cache_hit for {np.argwhere(S_b_temp[M:]==1).squeeze()}, path {path_n}') if debug_cache else 0
                chache_hits += 1
                delay_temp = delay_cache[S_id_edge_temp]
                np.copyto(Rcpu_temp,Rcpu_cache[S_id_edge_temp])
                np.copyto(Rmem_temp,Rmem_cache[S_id_edge_temp])
                delay_increase_temp = delay_temp - delay_new    # delay increase wrt the new state
                delay_increase_abs = delay_temp - delay_old 
                if delay_increase_abs > delay_increase_target:
                    # removing this dependency path will increase the delay too much
                    continue
                if skip_delay_increase and delay_increase_temp>0:
                    continue
            else:
                Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M))    # instance-set call frequency matrix of the temp state
                Nci_temp = computeNc(Fci_temp, M, 2)    # number of instance call per user request of the temp state
                delay_temp = computeDnTot(S_b_temp, Nci_temp, Fci_temp, Rs, RTT, Ne, lambd, M) # Total delay of the temp state. It includes only network delays
                delay_increase_temp = delay_temp - delay_new  # delay increase wrt the new state
                delay_increase_abs = delay_temp - delay_old # delay increase wrt the old state
                if delay_increase_abs > delay_increase_target:
                    # removing this dependency path will increase the delay too much
                    continue
                if skip_delay_increase and delay_increase_temp>0:
                    continue
                
                # compute the cost decrease removing this dependency path 
                # assumption is that cloud resource are reduce proportionally with respect to the reduction/increase of the number of times instances are called
                np.copyto(Rcpu_temp,Rcpu_old) 
                np.copyto(Rmem_temp,Rmem_old) 
                cloud_cpu_increase = -(1-Nci_temp[:M]/Nci_old[:M]) * Rcpu_old[:M]   
                cloud_mem_increase = -(1-Nci_temp[:M]/Nci_old[:M]) * Rmem_old[:M]  
                cloud_cpu_increase[np.isnan(cloud_cpu_increase)] = 0
                cloud_mem_increase[np.isnan(cloud_mem_increase)] = 0
                cloud_cpu_increase[cloud_cpu_increase==-inf] = 0
                cloud_mem_increase[cloud_mem_increase==-inf] = 0
                Rcpu_temp[M:] = Rcpu_temp[M:] - cloud_cpu_increase # edge cpu increase
                Rmem_temp[M:] = Rmem_temp[M:] - cloud_mem_increase # edge mem increase
                Rcpu_temp[:M] = Rcpu_temp[:M] + cloud_cpu_increase # cloud cpu decrease
                Rmem_temp[:M] = Rmem_temp[:M] + cloud_mem_increase # cloud mem decrease
                delay_cache[S_id_edge_temp] = delay_temp
                Rcpu_cache[S_id_edge_temp] = Rcpu_temp.copy() 
                Rmem_cache[S_id_edge_temp] = Rmem_temp.copy()
                print(f'cache insert for {np.argwhere(S_b_temp[M:]==1).squeeze()}, path {path_n}') if debug_cache else 0
            if delay_increase_abs > delay_increase_target:
                # removing this dependency path will increase the delay too much
                continue
            if skip_delay_increase and delay_increase_temp>0:
                continue
            Cost_edge_temp = Cost_cpu_edge * np.sum(Rcpu_temp[M:]) + Cost_mem_edge * np.sum(Rmem_temp[M:]) # Total edge cost of the temp state
            cost_decrease_temp = Cost_edge_new - Cost_edge_temp # cost decrease wrt the new state
            
            # weighting
            if delay_increase_temp < 0:
                # removal provides delay decrease,  weighting favour both cost and delay increase
                w = 1e6 - cost_decrease_temp * 1000 * delay_increase_temp 
                skip_delay_increase = True
            else:
                w = cost_decrease_temp / (1000*delay_increase_temp)
            
            print(f'state {S_b_temp[M:]}, cost decrease {cost_decrease_temp}, delay increase {1000*delay_increase_temp}, weight {w}') if debug2 else 0

            if w > w_max:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                np.copyto(Rcpu_opt,Rcpu_temp)
                np.copyto(Rmem_opt,Rmem_temp)
                cost_decrease_opt = cost_decrease_temp
                delay_increase_opt = delay_increase_temp
                delay_opt = delay_temp
                w_max = w
                
        # Prune not considered dependency paths whose microservices are going to be not contained in the edge to accelerate computation
        PR = []
        for pr,path_b in enumerate(dependency_paths_b_residual):
            if np.sum(path_b) != np.sum(path_b * S_b_opt[M:]):
                # dependency path not fully included at edge
                PR.append(pr)
                # cache cleaning
                S_id_edge_temp = str(S2id(S_b_temp[M:]))
                if S_id_edge_temp in delay_cache:
                    del delay_cache[S_id_edge_temp]
                    del Rcpu_cache[S_id_edge_temp]
                    del Rmem_cache[S_id_edge_temp]
        dependency_paths_b_residual = [dependency_paths_b_residual[pr] for pr in range(len(dependency_paths_b_residual)) if pr not in PR ]
        
        # cache cleaning
        for S_id_edge_temp_s in list(delay_cache):
            # when a cached state do not have some edge microservice that are in new (opt) state, it will be never reused for computation
            S_id_edge_temp = id2S(int(S_id_edge_temp_s),2**M)   # binary encoded state of the edge
            if np.sum(S_b_opt[M:]) != np.sum(S_id_edge_temp * S_b_opt[M:]):
                    del delay_cache[S_id_edge_temp_s]
                    del Rcpu_cache[S_id_edge_temp_s]
                    del Rmem_cache[S_id_edge_temp_s]
            else:
                print(f"cached state {np.argwhere(np.array(S_id_edge_temp)==1).squeeze()}") if debug_cache else 0
        print(f"cache size {len(delay_cache)}") if debug_cache else 0
        print(f"cache hit prob. {chache_hits/len(r)}") if debug else 0
    
    n_rounds = 1
    # compute final values
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new = computeDnTot(S_b_new, Nci_new, Fci_new, Rs, RTT, Ne, lambd, M)
    delay_increase_new = delay_new - delay_old
    np.copyto(Rcpu_new,Rcpu_old) 
    np.copyto(Rmem_new,Rmem_old) 
    cloud_cpu_increase = (1-Nci_new[:M]/Nci_old[:M]) * Rcpu_old[:M]   
    cloud_mem_increase = (1-Nci_new[:M]/Nci_old[:M]) * Rmem_old[:M]  
    cloud_cpu_increase[np.isnan(cloud_cpu_increase)] = 0
    cloud_mem_increase[np.isnan(cloud_mem_increase)] = 0
    cloud_cpu_increase[cloud_cpu_increase==-inf] = 0
    cloud_mem_increase[cloud_mem_increase==-inf] = 0
    Rcpu_new[M:] = Rcpu_new[M:] - cloud_cpu_increase # edge cpu decrease
    Rmem_new[M:] = Rmem_new[M:] - cloud_mem_increase # edge mem decrease
    Rcpu_new[:M] = Rcpu_new[:M] + cloud_cpu_increase # cloud cpu increase
    Rmem_new[:M] = Rmem_new[:M] + cloud_mem_increase     # cloud mem increase
    Cost_edge_new = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
    cost_decrease_new = Cost_edge_old - Cost_edge_new

    
    return S_b_new[M:].astype(int), Cost_edge_new, delay_increase_new, cost_decrease_new, n_rounds
    
if __name__ == "__main__":
    # Define the input variables
    np.random.seed(150273)
    RTT = 0.0869    # RTT edge-cloud
    M = 30 # n. microservices
    delay_increase_target = 0.03    # requested delay reduction
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
    x = 5
    random_values = np.random.choice(range(l), size=x, replace=False)
    for j in random_values:
        S_edge_b = np.minimum(S_edge_b + dependency_paths_b[j],1)
    # set random values for CPU and memory requests
    Rcpu = (np.random.randint(32,size=M)+1) * Rcpu_quota
    Rcpu[M-1]=0   # user has no CPU request
    Rcpu = np.append(Rcpu, Rcpu)
    Rmem = np.zeros(2*M)
    Rcpu[M:] = Rcpu[M:] * S_edge_b # set to zero the CPU requests of the instances not at the edge
    Rmem[M:] = Rmem[M:] * S_edge_b # set to zero the memory requests of the instances not at the edge

    # Call the unoffload function
    unoffload(Rcpu, Rmem, Fcm, M, lambda_val, Rs, S_edge_b, delay_increase_target, RTT, Ne, Cost_cpu_edge, Cost_mem_edge, 1)