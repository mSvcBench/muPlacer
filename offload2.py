# pylint: disable=C0103, C0301

import datetime
import numpy as np
import networkx as nx
from computeNcMat import computeNcMat
from build_Fci import buildFci
from build_Fci import buildFcinew
from S2id import S2id
from delayMat import delayMat 
from delayMat import delayMatNcFci
from id2S import id2S
from numpy import inf
from computeDnTot import computeDnTot

np.seterr(divide='ignore', invalid='ignore')


def offload(Rcpu_curr, Rmem_curr, Fcm, M, lambd, Rs, app_edge, delta_mes, RTT, Ne,depth):
    #x = datetime.datetime.now().strftime('%d-%m_%H:%M:%S')
    #filename = f'offload_{x}.mat'
    #np.save(filename, arr=[Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT])


    ## INITIALIZE VARIABLES ##
    app_edge = np.append(app_edge, 1) # Add the user in app_edge vector (user is in the edge cluster)
    S_b_curr = np.concatenate((np.ones(int(M)), app_edge))
    S_b_curr[M-1] = 0  # User is not in the cloud
    Ce = np.inf # CPU capacity of edge datacenter
    Me = np.inf # Memory capacity of edge datacenter
    Rs = np.append(Rs, 0)  # Add the user in the Rs vector
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to fit the number of data centers
    Cost_cpu_edge = 1
    Cost_mem_edge = 1
    Rcpu_req = np.tile(np.zeros(int(M)), 2)  # Seconds of CPU per request (set to zero for all microservices)
    Rcpu_req[M-1] = 0   
    Rcpu_req[2*M-1] = 0
    # Seconds of CPU per request for the user


    # SAVE CURRENT VALUES FOR METRICS ##
    ## compute instance-set call frequency matrix
    Fci = np.matrix(buildFcinew(S_b_curr, Fcm, M))
    Nci = computeNcMat(Fci, M, 2)
    
    #delay_curr = delayMatNcFci(S_b_curr, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci, Fci,2) # Delay of the current configuration
    delay_curr = computeDnTot(S_b_curr, Nci, Fci, Rs, RTT, Ne, lambd, M)
  
    Rcpu_edge_curr_sum = np.sum(S_b_curr[M:] * Rcpu_curr[M:]) # Total CPU requested by instances in the edge
    Rmem_edge_curr_sum = np.sum(S_b_curr[M:] * Rmem_curr[M:]) # Total Memory requested by instances in the edge
    Cost_cpu_edge_curr_sum = Cost_cpu_edge * Rcpu_edge_curr_sum # Total CPU cost
    Cost_mem_edge_curr_sum = Cost_mem_edge * Rmem_edge_curr_sum # Total Mem cost
    Cost_edge_curr = Cost_cpu_edge_curr_sum + Cost_mem_edge_curr_sum # Total cost

    ## SEARCH PATHS FROM USER TO INSTANCES ##
    G = nx.DiGraph(Fcm) # Create the microservice dependency graph 
    dependency_paths_b = np.empty((0,M), int) # binary encoded dependency paths
   

    ## OFFLOAD ##
    if delta_mes > 0:
        ## COMPUTE CLOUD ONLY DEPENDENCY PATHS ##
        for ms in range(M-1):
            # Find all dependency paths in the graph from user to microservice ms
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            # Check if the path is "valid"
            for path_n in paths_n:
                edge_presences = S_b_curr[M:][path_n]
                # If all microservices in the path have edge_presences == 1, this path is not "valid" because all microservices are running in the edge
                if all(value == 1 for value in edge_presences):
                    break
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
        delta_target = delta_mes
    ## UNOFFLOAD ##
    else:
        ## COMPUTE EDGE DEPENDENCY PATHS ##
        for ms in range(M-1):
            # Find all dependency paths in the graph from user to microservice ms
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            # Check if the path is "valid"
            for path_n in paths_n:
                edge_presences = S_b_curr[M:][path_n]
                # If not all microservices in the path have edge_presences == 1 this path is not "valid"
                if not all(value == 1 for value in edge_presences):
                    break
                else:
                    dependency_paths_b.extend(path_n)
                    path_b = np.zeros(M)  
                    path_b[path_n] = 1 # Binary-based encoding of the dependency path
        
        # Reset status with no microservice at the edge
        S_b_curr=np.ones(2*M)
        S_b_curr[M-1:2*M-1] = 0
        Sold_edge_b = S_b_curr[M:2*M] # For unoffload, the old status has no edge microservice
        S_edge_id_curr = S2id(Sold_edge_b) 
        #delay_target = delay_curr - delta_mes # Analitical target delay (SLO)
        #delta_target = delayMatNcFci(Scurr_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci, 2) - delay_target # For unoffload, the delta_target is from the status with no microservice at the edge  
        delta_target = delta_mes

    ## GREEDY ADDITION OF SUBGRAPHS TO EDGE CLUSTER ##
    
    dependency_paths_b_residual = dependency_paths_b.copy() # \Pi_r of paper
    S_b_opt = S_b_curr  # Inizialize the new edge status
    S_b_temp = np.zeros(2*M)
    S_b_new = np.zeros(2*M)
    Rcpu_opt = Rcpu_curr.copy()
    Rmem_opt = Rmem_curr.copy()
    Rcpu_new = np.zeros(2*M)
    Rmem_new = np.zeros(2*M)
    Rcpu_temp = np.zeros(2*M)
    Rmem_temp = np.zeros(2*M)
    delay_opt = delay_curr

    delay_cache=dict()
    Rcpu_cache=dict()
    Rmem_cache=dict()

    debug = True
    debug2 = False
    debug_cache = False
    skip_neg = False
    delta_cost_opt=0
    delta_opt=1
    

    while True:
        print(f'-----------------------') if debug else 0
        w_min = float("inf") # Initialize the weight
        skip_neg = False
        np.copyto(S_b_new,S_b_opt)
        np.copyto(Rcpu_new,Rcpu_opt)
        np.copyto(Rmem_new,Rmem_opt)
        delay_new = delay_opt # Delay of the new placement state
        Cost_edge_new  = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
        print(f'new ms {np.argwhere(S_b_new[M:]==1).squeeze()}, delta_delay {1000*(delay_curr-delay_new)}, cost {Cost_edge_new}, delta_cost/delta_delay {delta_cost_opt/(1000*delta_opt)}') if debug else 0
        
        # Check if the delay reduction is enough
        if delay_curr-delay_new >= delta_target:
            #delay reduction reached
            break
        # remove too long dependency paths
        print(f"depencency path no depth: {len(dependency_paths_b_residual)}") if debug else 0
        r = np.argwhere(np.sum(np.maximum(dependency_paths_b_residual-S_b_new[M:],0),axis=1)<=depth).squeeze()
        print(f"depencency path with depth: {len(r)}") if debug else 0
        if len(dependency_paths_b_residual) == 0:
            # All dependency path considered no other way to reduce delay
            break
        chache_hits = 0
        for i in r :
            path_b = dependency_paths_b_residual[i]
            # merging path_b and S_b_new into S_b_temp
            path_n = np.argwhere(path_b==1).squeeze()
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[M+path_n] = 1
            S_id_edge_temp=str(S2id(S_b_temp[M:]))
            # if False:
            if S_id_edge_temp in delay_cache:
                print(f'cache_hit for {np.argwhere(S_b_temp[M:]==1).squeeze()}, path {path_n}') if debug_cache else 0
                chache_hits += 1
                delay_temp = delay_cache[S_id_edge_temp]
                np.copyto(Rcpu_temp,Rcpu_cache[S_id_edge_temp])
                np.copyto(Rmem_temp,Rmem_cache[S_id_edge_temp])
                delta_delay = delay_new - delay_temp
                if skip_neg and delta_delay<0:
                    continue
            else:
                Fci_temp = np.matrix(buildFcinew(S_b_temp, Fcm, M))
                Nci_temp = computeNcMat(Fci_temp, M, 2)
                #delay_temp = delayMatNcFci(S_b_temp, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_temp, Fci_temp, 2) # Delay of the new placement state
                delay_temp = computeDnTot(S_b_temp, Nci_temp, Fci_temp, Rs, RTT, Ne, lambd, M)
                delta_delay = delay_new - delay_temp
                if skip_neg and delta_delay<0:
                    continue
                
                # compute the cost increase adding this dependency path 
                np.copyto(Rcpu_temp,Rcpu_curr) 
                np.copyto(Rmem_temp,Rmem_curr) 
                cloud_cpu_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rcpu_curr[:M]   
                cloud_mem_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rmem_curr[:M]  
                cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
                cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
                cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
                cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
                Rcpu_temp[M:] = Rcpu_temp[M:] + cloud_cpu_reduction # edge cpu increase
                Rmem_temp[M:] = Rmem_temp[M:] + cloud_mem_reduction # edge mem increase
                Rcpu_temp[:M] = Rcpu_temp[:M] - cloud_cpu_reduction # cloud cpu decrease
                Rmem_temp[:M] = Rmem_temp[:M] - cloud_mem_reduction     # cloud mem decrease
                delay_cache[S_id_edge_temp] = delay_temp
                Rcpu_cache[S_id_edge_temp] = Rcpu_temp.copy() 
                Rmem_cache[S_id_edge_temp] = Rmem_temp.copy()
                print(f'cache store for {np.argwhere(S_b_temp[M:]==1).squeeze()}, path {path_n}') if debug_cache else 0
            Cost_edge_temp = Cost_cpu_edge * np.sum(Rcpu_temp[M:]) + Cost_mem_edge * np.sum(Rmem_temp[M:]) # Total edge cost
            delta_cost = Cost_edge_temp - Cost_edge_new
            
            # weighting
            r_delta = delta_target - (delay_curr-delay_new) # residul delay to decrease wrt previous conf
            if delta_delay < 0:
                w = 1e6 - delta_cost * 1000 * delta_delay 
            else:
                w = delta_cost /  min(1000*delta_delay, 1000*r_delta)
                skip_neg = True
            
            print(f'state {S_b_temp[M:]}, cost {delta_cost}, delta_delay {1000*delta_delay}, weight {w}') if debug2 else 0

            if w < w_min:
                np.copyto(S_b_opt,S_b_temp)
                np.copyto(Rcpu_opt,Rcpu_temp)
                np.copyto(Rmem_opt,Rmem_temp)
                delta_cost_opt = delta_cost
                delta_opt = delta_delay
                delay_opt = delay_temp
                w_min = w
                
        # Prune not considered dependency paths whose microservices are going to be contained in the edge
        PR = []
        for pr,path_b in enumerate(dependency_paths_b_residual):
            if np.sum(path_b) == np.sum(path_b * S_b_opt[M:]):
                # dependency path already fully included at edge
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
            # when a cached state do not have edge microservice in new (opt) state it will be never reused for calculos
            S_id_edge_temp = id2S(int(S_id_edge_temp_s),2**M)
            if np.sum(S_b_opt[M:]) != np.sum(S_id_edge_temp * S_b_opt[M:]):
                    del delay_cache[S_id_edge_temp_s]
                    del Rcpu_cache[S_id_edge_temp_s]
                    del Rmem_cache[S_id_edge_temp_s]
            else:
                print(f"cached state {np.argwhere(np.array(S_id_edge_temp)==1).squeeze()}") if debug_cache else 0
        print(f"cache size {len(delay_cache)}") if debug_cache else 0
        print(f"cache hit prob. {chache_hits/len(r)}") if debug else 0
        

    # cleaning phase
    while True:
        c_max=0
        i_max=-1
        for i in range(M, 2*M-1):
            if S_b_new[i]==1:
                # try remove microservice
                np.copyto(S_b_temp,S_b_new)
                S_b_temp[i] = 0
                delta_final2 = delay_curr - delayMat(S_b_temp, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # delay delta reached
                if delta_final2>=delta_target:
                    # possible removal
                    if Rcpu_new[i]*Cost_cpu_edge > c_max:
                        i_max = i
                        c_max = Rcpu_new[i]*Cost_cpu_edge
        if i_max>-1:
            print('cleaning')
            S_b_new[i_max] = 0
            Cost_edge_new = Cost_edge_new - Rcpu_new[i_max]*Cost_cpu_edge
        else:
            break
            
    
    n_rounds = 1

    # compute final values
    Fci_new = np.matrix(buildFcinew(S_b_new, Fcm, M))
    Nci_new = computeNcMat(Fci_new, M, 2)
    #delay_new = delayMatNcFci(S_b_new, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_new, Fci_new, 2) # Delay of the new placement state
    delay_new = computeDnTot(S_b_new, Nci_new, Fci_new, Rs, RTT, Ne, lambd, M)
    delta_new = delay_curr - delay_new
    np.copyto(Rcpu_new,Rcpu_curr) 
    np.copyto(Rmem_new,Rmem_curr) 
    cloud_cpu_reduction = (1-Nci_new[:M]/Nci[:M]) * Rcpu_curr[:M]   
    cloud_mem_reduction = (1-Nci_new[:M]/Nci[:M]) * Rmem_curr[:M]  
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
    cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
    Rcpu_new[M:] = Rcpu_new[M:] + cloud_cpu_reduction # edge cpu increase
    Rmem_new[M:] = Rmem_new[M:] + cloud_mem_reduction # edge mem increase
    Rcpu_new[:M] = Rcpu_new[:M] - cloud_cpu_reduction # cloud cpu decrease
    Rmem_new[:M] = Rmem_new[:M] - cloud_mem_reduction     # cloud mem decrease
    Cost_edge_new = Cost_cpu_edge * np.sum(Rcpu_new[M:]) + Cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
    delta_cost = Cost_edge_new - Cost_edge_curr 

    
    return S_b_new[M:].astype(int).tolist(), Cost_edge_new, delta_new, delta_cost, n_rounds
