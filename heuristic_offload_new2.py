import numpy as np
import networkx as nx
from S2id import S2id
from delayMat import delayMat 
from id2S import id2S
import matplotlib.pyplot as plt


#   RTT : edge-cloud Round Trip Time 
#   Ne : edge-cloud bit rate
#   lambd : average user request frequency
#   Rs : response size in bytes of each instance-set
#   Fcm : call frequency matrix
#   Nc : average number of time the instance-set is involved per user request 
#   M : number of microservices
#   Rcpu_req : CPU seconds for internal functions
#   Ce, Me : CPU and Memory of the edge
#   Rcpu, Rmem : CPU and Mem requested by microservices
#   db = 0 if no db
#   e : number of datacenters
#   app_edge: microservices already in the edge cluster
#   min_delay_delta: minimum delay reduction

# _b : binary encoding of a set of nodes/services 
# _id : identifier of a set of nodes come out from binary encoding
# _n : identifiers of the nodes of a set

def heuristic_offload(Fcm, RTT, Rcpu_req, Rcpu, Rmem, Cost_cpu_edge, Cost_mem_edge, Ce, Me, Ne, lambd, Rs, M, db, horizon, e, Sold_b, delta_mes):
    ## SEARCH ALL PATHS FROM USER TO INSTANCES ##


    # save current value for metrics
    Scurr_edge_b = Sold_b[M:2*M] # Binary placement status containing edge microservices only
    Scurr_edge_id = S2id(Scurr_edge_b) # id-based placement status containing only edge microservices
    delay_curr = delayMat(Sold_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2)
    Rcpu_edge_curr = Rcpu[M:]
    Rmem_edge_curr = Rmem[M:]
    Rcpu_edge_sum_curr = np.sum(Scurr_edge_b * Rcpu_edge_curr) # Total CPU requested by instances in the edge
    Rmem_edge_sum_curr = np.sum(Scurr_edge_b * Rmem_edge_curr) # Total Memory requested by instances in the edge
    Cost_cpu_edge_sum_curr = Cost_cpu_edge * Rcpu_edge_sum_curr # Total CPU cost
    Cost_mem_edge_sum_curr = Cost_mem_edge * Rmem_edge_sum_curr # Total Mem cost
    Cost_edge_curr = Cost_cpu_edge_sum_curr + Cost_mem_edge_sum_curr
    
    
    G = nx.DiGraph(Fcm) # Create the microservice dependency graph 
    dependency_paths_set = [] # cloud only dependency paths, each path is a list of the ids (0,M-1) of the contained microservices. \Pi_c of paper
    dependency_paths_set_id = [] # id-based cloud only dependency paths

    if (delta_mes > 0):
        # offload
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
                    dependency_path_set_b[path] = 1 # binary-based encoding of the dependency path
                    dependency_path_set_id = S2id(dependency_path_set_b)  # id-based encoding of the dependency path
                    dependency_paths_set_id.append(dependency_path_set_id) # \Pi_c of paper
        delay_target = delay_curr - delta_mes # analitical target delay 
        delta_target = delay_curr - delay_target # targeted delay increase
        Sold_edge_b = Scurr_edge_b # For offload, the old status is the current one
        Sold_edge_id = Scurr_edge_id 
    else:
        ## COMPUTE EDGE DEPENDENCY PATHS ##
        # Find all dependency paths in the graph from user to microservice ms
        for ms in range(M-1):
            paths = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
            # Check if the path is "valid"
            for path in paths:
                edge_presences = Scurr_edge_b[path]
                # If all microservices in the path have edge_presences == 1, this path is "valid" because all microservices are running in the edge
                if not all(value == 1 for value in edge_presences):
                    break
                else:
                    dependency_paths_set.extend(path)
                    dependency_path_set_b = np.zeros(M)  
                    dependency_path_set_b[path] = 1 # binary-based encoding of the dependency path
                    dependency_path_set_id = S2id(dependency_path_set_b)  # id-based encoding of the dependency path
                    dependency_paths_set_id.append(dependency_path_set_id) # \Pi_c of paper
        
        # reset status with no microservice at the edge
        Sold_b=np.ones(2*M)
        Sold_b[M-1:2*M-1] = 0
        Sold_edge_b = Sold_b[M:2*M] # For unoffload, the old status has no edge microservice
        Sold_edge_id = S2id(Sold_edge_b) 
        delay_target = delay_curr - delta_mes # analitical target delay 
        delta_target = delayMat(Sold_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) - delay_target # for unoffload, the delta_target is from the status with no microservice at the edge  
    
    ## GREEDY ADDITION OF SUBGRAPHS TO EDGE CLUSTER ##
    
    delay_old = delayMat(Sold_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the original configuration
    Rcpu_edge = Rcpu[M:]
    Rmem_edge = Rmem[M:]
    Rcpu_old_edge_sum = np.sum(Sold_edge_b * Rcpu_edge) # Total CPU requested by instances in the edge
    Rmem_old_edge_sum = np.sum(Sold_edge_b * Rmem_edge) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_old_edge_sum # Total CPU cost
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_old_edge_sum # Total Mem cost
    
    dependency_paths_cloud_only_r_id = dependency_paths_set_id.copy() # \Pi_r of paper
    Snew_edge_id = Sold_edge_id
        
    while True:
        w_min = float("inf")
        Sopt_id = Snew_edge_id
        Snew_edge_b = np.array(id2S(Snew_edge_id,2**M))
        Snew_b = np.ones(2*M)
        Snew_b[M-1]=0
        Snew_b[M:] = Snew_edge_b
        Rcpu_new_edge_sum = np.sum(Snew_edge_b * Rcpu_edge) # CPU requested by the new state
        Rmem_new_edge_sum = np.sum(Snew_edge_b * Rmem_edge) # Memory requested by the new state
        Cost_cpu_new_edge_sum = Cost_cpu_edge * Rcpu_new_edge_sum # Total CPU cost
        Cost_mem_new_edge_sum = Cost_mem_edge * Rmem_new_edge_sum # Total Mem cost  
        Cost_opt = Cost_cpu_new_edge_sum + Cost_mem_new_edge_sum 
        delay_new = delayMat(Snew_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the new placement state
        r_delta_delay = delta_target - (delay_old-delay_new)
        # Check if the delay reduction is enough
        if r_delta_delay <= 0:
            #delay reduction reached
            break
        for path_id in dependency_paths_cloud_only_r_id :
            S_edge_temp_id = Snew_edge_id # Starting configuration
            S_edge_temp_id = np.bitwise_or(path_id - 1, Snew_edge_id - 1) + 1  # New edge state adding new dependency path
            S_edge_temp_b = np.array(id2S(S_edge_temp_id, 2 ** M)) # New edge state in binary encoding
            Rcpu_temp_sum = np.sum(S_edge_temp_b * Rcpu_edge) # CPU requested by the new state
            Rmem_temp_sum = np.sum(S_edge_temp_b * Rmem_edge) # Memory requested by the new state
            Cost_cpu_edge_temp_sum = Cost_cpu_edge * Rcpu_temp_sum # Total CPU cost
            Cost_mem_edge_temp_sum = Cost_mem_edge * Rmem_temp_sum # Total Mem cost   
            S_temp_b = Sold_b
            S_temp_b[M:] = S_edge_temp_b
            delay_temp = delayMat(S_temp_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the new placement state
            delta_delay = delay_new - delay_temp
            delta_cost = (Cost_cpu_edge_temp_sum-Cost_cpu_new_edge_sum) + (Cost_mem_edge_temp_sum-Cost_mem_new_edge_sum)
            w = delta_cost / min(delta_delay, r_delta_delay)
            if (w < w_min and Rcpu_temp_sum <= Ce and Rmem_temp_sum <= Me and delta_delay>0):
                Sopt_id = S_edge_temp_id
                Cost_opt = Cost_cpu_edge_temp_sum + Cost_mem_edge_temp_sum  # cost of the solution
                w_min = w

        if (Sopt_id == Snew_edge_id):
            # no additional delay reduction possible
            break
        else:
            Snew_edge_id = Sopt_id
            # Prune not considered dependency graph whose microservices are already contained in the edge
            PR = []
            for pr in range(len(dependency_paths_cloud_only_r_id)):
                if np.bitwise_and(dependency_paths_cloud_only_r_id[pr] - 1, Snew_edge_id - 1) + 1 == dependency_paths_cloud_only_r_id[pr]:
                    # dependency path already fully included at edge
                    PR.append(pr)
            dependency_paths_cloud_only_r_id = [dependency_paths_cloud_only_r_id[pr] for pr in range(len(dependency_paths_cloud_only_r_id)) if pr not in PR ]
            if len(dependency_paths_cloud_only_r_id) == 0:
                # All dependency path considered
                break

    S_new_edge_b = id2S(int(Snew_edge_id), 2 ** M)
    S_new_b = np.ones(2*M)
    S_new_b[M-1]=0
    S_new_b[M:2*M] = S_new_edge_b   # new edge binary status
    delta_cost_opt = Cost_opt - Cost_edge_curr  # cost variation
    delta_final = delay_curr - delayMat(S_new_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # delay delta reached
    return S_new_edge_b, Cost_opt, delta_final, delta_cost_opt