# pylint: disable=C0103, C0301

import datetime
import numpy as np
import networkx as nx
from computeNcMat import computeNcMat
from buildFci import buildFci
from buildFci import buildFcinew
from S2id import S2id
from delayMat import delayMat 
from delayMat import delayMatNcFci
from id2S import id2S
from numpy import inf


def offload(Rcpu, Rmem, Fcm, M, lambd, Rs, app_edge, delta_mes, RTT, Ne,depth):
    #x = datetime.datetime.now().strftime('%d-%m_%H:%M:%S')
    #filename = f'offload_{x}.mat'
    #np.save(filename, arr=[Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT])


    ## INITIALIZE VARIABLES ##
    app_edge = np.append(app_edge, 1) # Add the user in app_edge vector (user is in the edge cluster)
    S_b_curr = np.concatenate((np.ones(int(M)), app_edge))
    S_b_curr[M-1] = 0  # User is not in the cloud
    e = 2  # Number of datacenters
    Ubit = np.arange(1, e+1) * M  # User position in the state vector
    Ce = np.inf # CPU capacity of edge datacenter
    Me = np.inf # Memory capacity of edge datacenter
    Rs = np.append(Rs, 0)  # Add the user in the Rs vector
    Rs = np.tile(Rs, e)  # Expand the Rs vector to fit the number of data centers
    Cost_cpu_edge = 1
    Cost_mem_edge = 1
    Rcpu_req = np.tile(np.zeros(int(M)), e)  # Seconds of CPU per request (set to zero for all microservices)

    # Seconds of CPU per request for the user
    Rcpu_req[int(Ubit[0])-1] = 0   
    Rcpu_req[int(Ubit[1])-1] = 0

    # SAVE CURRENT VALUES FOR METRICS ##
    ## compute instance-set call frequency matrix
    #Fci = np.matrix(buildFci(Scurr_b, Fcm, M, e))
    Fci = np.matrix(buildFcinew(S_b_curr, Fcm, M))
    Nci = computeNcMat(Fci, M, e)
    
    delay_curr = delayMatNcFci(S_b_curr, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci, Fci,2) # Delay of the current configuration
  
    Scurr_edge_b = S_b_curr[M:2*M] # Binary placement status containing only edge microservices
    S_edge_id_curr = S2id(Scurr_edge_b) # id-based placement status containing only edge microservices
    Rcpu_edge_curr = Rcpu[M:] # CPU requested by the edge microservices
    Rmem_edge_curr = Rmem[M:] # Memory requested by the edge microservices
    Rcpu_edge_curr_sum = np.sum(Scurr_edge_b * Rcpu_edge_curr) # Total CPU requested by instances in the edge
    Rmem_edge_curr_sum = np.sum(Scurr_edge_b * Rmem_edge_curr) # Total Memory requested by instances in the edge
    Cost_cpu_edge_curr_sum = Cost_cpu_edge * Rcpu_edge_curr_sum # Total CPU cost
    Cost_mem_edge_curr_sum = Cost_mem_edge * Rmem_edge_curr_sum # Total Mem cost
    Cost_edge_curr = Cost_cpu_edge_curr_sum + Cost_mem_edge_curr_sum # Total cost

    Scurr_cloud_b = S_b_curr[0:M] # Binary placement status containing only edge microservices
    Scurr_cloud_id = S2id(Scurr_cloud_b) # id-based placement status containing only edge microservices
    Rcpu_cloud_curr = Rcpu[:M] # CPU requested by the edge microservices
    Rmem_cloud_curr = Rmem[:M] # Memory requested by the edge microservices
    # Rcpu_cloud_curr_sum = np.sum(Scurr_cloud_b * Rcpu_cloud_curr) # Total CPU requested by instances in the edge
    # Rmem_cloud_curr_sum = np.sum(Scurr_cloud_b * Rmem_cloud_curr) # Total Memory requested by instances in the edge
    # Cost_cpu_cloud_curr_sum = Cost_cpu_edge * Rcpu_cloud_curr_sum # Total CPU cost
    # Cost_mem_cloud_curr_sum = Cost_mem_edge * Rmem_cloud_curr_sum # Total Mem cost
    # Cost_cloud_curr = Cost_cpu_cloud_curr_sum + Cost_mem_cloud_curr_sum # Total cost

    ## SEARCH PATHS FROM USER TO INSTANCES ##
    G = nx.DiGraph(Fcm) # Create the microservice dependency graph 
    dependency_paths_set = [] # Cloud only dependency paths, each path is a list of the ids (0,M-1) of the contained microservices. \Pi_c of paper
    dependency_paths_set_id = [] # id-based cloud only dependency paths

    ## OFFLOAD ##
    if delta_mes > 0:
        ## COMPUTE CLOUD ONLY DEPENDENCY PATHS ##
        for ms in range(M-1):
            # Find all dependency paths in the graph from user to microservice ms
            paths = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            # Check if the path is "valid"
            for path in paths:
                edge_presences = Scurr_edge_b[path]
                # If all microservices in the path have edge_presences == 1, this path is not "valid" because all microservices are running in the edge
                if all(value == 1 for value in edge_presences):
                    break
                else:
                    dependency_paths_set.extend(path)
                    dependency_path_set_b = np.zeros(M)
                    dependency_path_set_b[path] = 1 # Binary-based encoding of the dependency path
                    dependency_path_set_id = S2id(dependency_path_set_b)  # id-based encoding of the dependency path
                    dependency_paths_set_id.append(dependency_path_set_id) # \Pi_c of paper
        ######### ?????????????????? #########
        delay_target = delay_curr - delta_mes # Analitical target delay (SLO)
        delta_target = delay_curr - delay_target # Targeted delay increase
        ######################################
        Sold_edge_b = Scurr_edge_b.copy() # For offload, the old status is the current one
        S_edge_id_curr = S_edge_id_curr # id of the old status

    ## UNOFFLOAD ##
    else:
        ## COMPUTE EDGE DEPENDENCY PATHS ##
        for ms in range(M-1):
            # Find all dependency paths in the graph from user to microservice ms
            paths = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            # Check if the path is "valid"
            for path in paths:
                edge_presences = Scurr_edge_b[path]
                # If not all microservices in the path have edge_presences == 1 this path is not "valid"
                if not all(value == 1 for value in edge_presences):
                    break
                else:
                    dependency_paths_set.extend(path)
                    dependency_path_set_b = np.zeros(M)  
                    dependency_path_set_b[path] = 1 # Binary-based encoding of the dependency path
                    dependency_path_set_id = S2id(dependency_path_set_b)  # id-based encoding of the dependency path
                    dependency_paths_set_id.append(dependency_path_set_id) # \Pi_c of paper
        
        # Reset status with no microservice at the edge
        S_b_curr=np.ones(2*M)
        S_b_curr[M-1:2*M-1] = 0
        Sold_edge_b = S_b_curr[M:2*M] # For unoffload, the old status has no edge microservice
        S_edge_id_curr = S2id(Sold_edge_b) 
        #delay_target = delay_curr - delta_mes # Analitical target delay (SLO)
        #delta_target = delayMatNcFci(Scurr_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci, 2) - delay_target # For unoffload, the delta_target is from the status with no microservice at the edge  
        delta_target = delta_mes

    ## GREEDY ADDITION OF SUBGRAPHS TO EDGE CLUSTER ##
    
    dependency_paths_cloud_only_r_id = dependency_paths_set_id.copy() # \Pi_r of paper
    S_edge_id_opt = S_edge_id_curr  # Inizialize the new edge status
    S_b_temp = np.empty_like(S_b_curr)
    S_b_new = np.empty_like(S_b_curr)
    Rcpu_edge_opt = Rcpu_edge_curr.copy()
    Rmem_edge_opt = Rmem_edge_curr.copy()
    Rcpu_edge_new = np.empty_like(Rcpu_edge_curr)
    Rmem_edge_new = np.empty_like(Rmem_edge_curr)
    Rcpu_edge_temp = np.empty_like(Rcpu_edge_curr)
    Rmem_edge_temp = np.empty_like(Rmem_edge_curr)
    Rcpu_cloud_temp = np.empty_like(Rcpu_cloud_curr)
    Rmem_cloud_temp = np.empty_like(Rmem_cloud_curr)
    delay_opt = delay_curr

    debug = False
    debug2 = False
    skip_neg = False
    delta_cost_opt=0
    delta_opt=1
    

    while True:
        print(f'-----------------------') if debug else 0
        w_min = float("inf") # Initialize the weight
        S_edge_id_new = S_edge_id_opt  # Initialize new status as the optimum one of the previous greedy round
        np.copyto(Rcpu_edge_new,Rcpu_edge_opt)
        np.copyto(Rmem_edge_new,Rmem_edge_opt)
        S_edge_b_new = np.array(id2S(S_edge_id_new,2**M)) # New edge status in binary encoding
        S_b_new = np.ones(2*M)
        S_b_new[M-1]=0
        S_b_new[M:] = S_edge_b_new[:]
        
        delay_new = delay_opt # Delay of the new placement state
        Cost_edge_new  = Cost_cpu_edge * np.sum(Rcpu_edge_new) + Cost_mem_edge * np.sum(Rmem_edge_new) # Total edge cost
        print(f'new subpath {S_edge_b_new}, delta_delay {1000*(delay_curr-delay_new)}, cost {Cost_edge_new}, delta_cost/delta_delay {delta_cost_opt/(1000*delta_opt)}') if debug else 0
        
        # Check if the delay reduction is enough
        if delay_curr-delay_new >= delta_target:
            #delay reduction reached
            break
        if len(dependency_paths_cloud_only_r_id) == 0:
            # All dependency path considered no other way to reduce delay
            break
        for path_id in dependency_paths_cloud_only_r_id :
            S_edge_id_temp = np.bitwise_or(path_id - 1, S_edge_id_new - 1) + 1  # New edge state adding new dependency path
            S_edge_b_temp = np.array(id2S(S_edge_id_temp, 2 ** M)) # New edge state in binary encoding
            if np.sum(S_edge_b_temp-S_edge_b_new)>depth:
                continue
            
            S_b_temp = S_b_new.copy()
            S_b_temp[M:] = S_edge_b_temp[:]

            #Fci_temp = np.matrix(buildFci(S_b_temp, Fcm, M, e))
            Fci_temp = np.matrix(buildFcinew(S_b_temp, Fcm, M))
            Nci_temp = computeNcMat(Fci_temp, M, e)
            delay_temp = delayMatNcFci(S_b_temp, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_temp, Fci_temp, 2) # Delay of the new placement state
            if skip_neg and delay_temp<0:
                continue
            delta_delay = delay_new - delay_temp
            
            # compute the cost increase adding this dependency path 
            np.copyto(Rcpu_edge_temp,Rcpu_edge_curr) # CPU requested by the edge microservices
            np.copyto(Rmem_edge_temp,Rmem_edge_curr) # Memory requested by the edge microservices
            np.copyto(Rcpu_cloud_temp,Rcpu_cloud_curr) # CPU requested by the edge microservices
            np.copyto(Rmem_cloud_temp,Rmem_cloud_curr) # Memory requested by the edge microservices
            cloud_cpu_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rcpu_cloud_curr  # equal to edge cpu increase
            cloud_mem_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rmem_cloud_curr  # equal to edge mem increase
            cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
            cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
            cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
            cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
            Rcpu_edge_temp = Rcpu_edge_temp + cloud_cpu_reduction
            Rmem_edge_temp = Rmem_edge_temp + cloud_mem_reduction
            Rcpu_cloud_temp = Rcpu_cloud_temp - cloud_cpu_reduction
            Rmem_cloud_temp = Rmem_cloud_temp - cloud_mem_reduction


            # for k in range(M-1):
            #     if Nci[k]>0:                
            #         cloud_cpu_reduction = (1-Nci_temp[k]/Nci[k]) * Rcpu_cloud_curr[k]  # equal to edge cpu increase
            #         cloud_mem_reduction = (1-Nci_temp[k]/Nci[k]) * Rmem_cloud_curr[k]  # equal to edge mem increase
            #         Rcpu_edge_temp[k] = Rcpu_edge_temp[k] + cloud_cpu_reduction
            #         Rmem_edge_temp[k] = Rmem_edge_temp[k] + cloud_mem_reduction
            #         Rcpu_cloud_temp[k] = Rcpu_cloud_temp[k] - cloud_cpu_reduction
            #         Rmem_cloud_temp[k] = Rmem_cloud_temp[k] - cloud_mem_reduction
            Cost_edge_temp = Cost_cpu_edge * np.sum(Rcpu_edge_temp) + Cost_mem_edge * np.sum(Rmem_edge_temp) # Total edge cost
            # delta_cost = Cost_edge_temp - Cost_edge_curr
            delta_cost = Cost_edge_temp - Cost_edge_new
            
            r_delta = delta_target - (delay_curr-delay_new) # residul delay to decrease wrt previous conf
            if delta_delay < 0:
                w = 1e6 - delta_cost / min(1000*delta_delay, 1000*r_delta) 
                #w = 1e6 - (delta_cost/delay_curr) / max((delta_target-delta_delay)/delta_target, 1e-3)
            else:
                w = delta_cost /  min(1000*delta_delay, 1000*r_delta)
                skip_neg = True
            
            print(f'subpath {S_edge_b_temp}, cost {delta_cost}, delta_delay {1000*delta_delay}, weight {w}') if debug2 else 0

            if w < w_min:
                S_edge_id_opt = S_edge_id_temp
                Rcpu_edge_opt = Rcpu_edge_temp.copy()
                Rmem_edge_opt = Rmem_edge_temp.copy()
                delta_cost_opt = delta_cost
                delta_opt = delta_delay
                delay_opt = delay_temp
                w_min = w
                
        if S_edge_id_opt == S_edge_id_new:
            # no additional delay reduction possible
            break
        else:
            S_edge_id_new = S_edge_id_opt
            # Prune not considered dependency graph whose microservices are already contained in the edge
            PR = []
            for pr,dependency_path_cloud_only_r_id in enumerate(dependency_paths_cloud_only_r_id):
                if np.bitwise_and(dependency_path_cloud_only_r_id - 1, S_edge_id_new - 1) + 1 == dependency_path_cloud_only_r_id:
                    # dependency path already fully included at edge
                    PR.append(pr)
            dependency_paths_cloud_only_r_id = [dependency_paths_cloud_only_r_id[pr] for pr in range(len(dependency_paths_cloud_only_r_id)) if pr not in PR ]
        

    
    # cleaning phase
    S_new_c = np.empty_like(S_b_new)
    while True:
        c_max=0
        i_max=-1
        for i in range(M, 2*M-1):
            if S_b_new[i]==1:
                # try remove microservice
                np.copyto(S_new_c,S_b_new)
                S_new_c[i] = 0
                delta_final2 = delay_curr - delayMat(S_new_c, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # delay delta reached
                if delta_final2>=delta_target:
                    # possible removal
                    if Rcpu_edge_new[i-M]*Cost_cpu_edge > c_max:
                        i_max = i
                        c_max = Rcpu_edge_new[i-M]*Cost_cpu_edge
        if i_max>-1:
            print('cleaning')
            S_b_new[i_max] = 0
            Cost_edge_new = Cost_edge_new - Rcpu_edge_new[i_max-M]*Cost_cpu_edge
        else:
            break
            
    

    #Final cost computation
    Fci_new = np.matrix(buildFcinew(S_b_new, Fcm, M))
    Nci_new = computeNcMat(Fci_new, M, e)
    delay_final = delayMatNcFci(S_b_temp, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_new, Fci_new, 2) # Delay of the new placement state
    delta_final = delay_curr - delay_final
    
    # compute the cost increase adding this dependency path 
    np.copyto(Rcpu_edge_new,Rcpu_edge_curr) # CPU requested by the edge microservices
    np.copyto(Rmem_edge_new,Rmem_edge_curr) # Memory requested by the edge microservices
    cloud_cpu_reduction = (1-Nci_new[:M]/Nci[:M]) * Rcpu_cloud_curr  # equal to edge cpu increase
    cloud_mem_reduction = (1-Nci_new[:M]/Nci[:M]) * Rmem_cloud_curr  # equal to edge mem increase
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
    cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
    Rcpu_edge_new = Rcpu_edge_new + cloud_cpu_reduction
    Rmem_edge_temp = Rmem_edge_temp + cloud_mem_reduction
    Cost_edge_temp = Cost_cpu_edge * np.sum(Rcpu_edge_new) + Cost_mem_edge * np.sum(Rmem_edge_new) # Total edge cost

    delta_cost_opt = Cost_edge_new - Cost_edge_curr  # cost variation
    #print(S_new_edge_b)
    #print(Cost_edge_new)
    delta_final = delay_curr - delayMat(S_new_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # delay delta reached
    n_rounds = 1
    
    return S_new_edge_b, Cost_edge_new, delta_final, delta_cost_opt, n_rounds
