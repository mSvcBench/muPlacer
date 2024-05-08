import numpy as np
import networkx as nx
from S2id import S2id
from old.delayMat import delayMat 
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

def heuristic_unoffload(Fcm, RTT, Rcpu_req, Rcpu, Rmem, Cost_cpu_edge, Cost_mem_edge, Ce, Me, Ne, lambd, Rs, M, db, horizon, e, Sold_b, delta_mes):
    
    delta_unoff = -delta_mes
    ## SEARCH ALL PATHS FROM USER TO INSTANCES ##
    G = nx.DiGraph(Fcm) # Create the microservice dependency graph 

    #nx.draw_planar(G,with_labels=True)
    #plt.show()
    
    Sold_edge_b = Sold_b[M:2*M] # Binary placement status containing edge microservices only
    Sold_edge_id = S2id(Sold_edge_b) # id-based placement status containing only edge microservices
    
    dependency_paths_edge = [] # cloud only dependency paths, each path is a list of the ids (0,M-1) of the contained microservices. \Pi_c of paper
    dependency_paths_edge_id = [] # id-based cloud only dependency paths

    ## COMPUTE EDGE DEPENDENCY PATHS ##
    for ms in range(M-1):
        # Find all dependency paths in the graph from user to microservice ms
        paths = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        # Check if the path is "valid"
        for path in paths:
            edge_presences = Sold_edge_b[path]
            # If all microservices in the path have edge_presences == 1, this path is not "valid" because all microservices are running in the edge
            if not all(value == 1 for value in edge_presences):
                break
            else:
                dependency_paths_edge.extend(path)
                dependency_path_edge_b = np.zeros(M)  
                dependency_path_edge_b[path] = 1 # binary-based encoding of the dependency path
                dependency_path_edge_id = S2id(dependency_path_edge_b)  # id-based encoding of the dependency path
                dependency_paths_edge_id.append(dependency_path_edge_id)  # \Pi_e of paper
    
    
    ## GREEDY ADDITION OF SUBGRAPHS TO EDGE CLUSTER ##
    
    delay_old = delayMat(Sold_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the original configuration
    Rcpu_edge = Rcpu[M:]
    Rmem_edge = Rmem[M:]
    Rcpu_old_edge_sum = np.sum(Sold_edge_b * Rcpu_edge) # Total CPU requested by instances in the edge
    Rmem_old_edge_sum = np.sum(Sold_edge_b * Rmem_edge) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_old_edge_sum # Total CPU cost
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_old_edge_sum # Total Mem cost
    
    dependency_paths_edge_r_id = dependency_paths_edge_id.copy() # \Pi_r of paper
    Snew_edge_id = Sold_edge_id
    
    while True:
        w_min = 1e6
        Sopt_id = Snew_edge_id
        delta_delay_opt = -1
        for path_id in dependency_paths_edge_r_id :
            dependency_paths_edge_r_id_without_path_id =  dependency_paths_edge_r_id.copy() 
            dependency_paths_edge_r_id_without_path_id.remove(path_id)
            S_edge_temp_id = state_id_from_dependency_paths_id(dependency_paths_edge_r_id_without_path_id,M) # \Pi_r \ \pi of the paper
            S_edge_temp_b = np.array(id2S(S_edge_temp_id, 2 ** M)) # New edge state in binary encoding
            Rcpu_temp_sum = np.sum(S_edge_temp_b * Rcpu_edge) # CPU requested by the new state
            Rmem_temp_sum = np.sum(S_edge_temp_b * Rmem_edge) # Memory requested by the new state
            Cost_cpu_edge_temp_sum = Cost_cpu_edge * Rcpu_temp_sum # Total CPU cost
            Cost_mem_edge_temp_sum = Cost_mem_edge * Rmem_temp_sum # Total Mem cost   
            S_temp_b = Sold_b
            S_temp_b[M:] = S_edge_temp_b
            delay_temp = delayMat(S_temp_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the new placement state
            delta_delay = delay_temp - delay_old # delay increase with new config
            delta_cost = (Cost_cpu_edge_old_sum-Cost_cpu_edge_temp_sum) + (Cost_mem_edge_old_sum-Cost_mem_edge_temp_sum) # cost rediction with new config
            w = delta_cost / delta_delay
            if (delta_delay>delta_mes and w < w_min and Rcpu_temp_sum <= Ce and Rmem_temp_sum <= Me ):
                Sopt_id = S_edge_temp_id
                delta_delay_opt = delta_delay
                w_min = w
            
        if (Sopt_id == Snew_edge_id):
            # no additional delay increase possible
            break
        else:
            if delta_delay_opt < delta_mes:
                # too much delay increase, exit and get the Snew_edge_id computed before
                break
            else:
                Snew_edge_id = Sopt_id
                # Prune not considered dependency graph whose microservices are already contained in the edge
                PR = []
                for pr in range(len(dependency_paths_edge_r_id)):
                    if np.bitwise_and(dependency_paths_edge_r_id[pr] - 1, Snew_edge_id - 1) + 1 == dependency_paths_edge_r_id[pr]:
                        # dependency path already fully included at edge
                        PR.append(pr)
                dependency_paths_edge_r_id = [dependency_paths_edge_r_id[pr] for pr in range(len(dependency_paths_edge_r_id)) if pr not in PR ]
                if len(dependency_paths_edge_r_id) == 0:
                    # All dependency path considered
                    break

    S_new_edge_b = id2S(int(Snew_edge_id), 2 ** M)
    return S_new_edge_b

def state_id_from_dependency_paths_id(dependency_paths_id,M):
    S_b = np.zeros(M)
    for path_id in dependency_paths_id:
        path_b = id2S(path_id,2**M)
        path = np.nonzero(path_b)
        S_b[path] = 1
    S_b[M-1] = 1  # user can not be removed
    return S2id(S_b)

def main():
    dependency_paths_id=[]
    dependency_paths_id.append(2)
    dependency_paths_id.append(4)
    state_id_from_dependency_paths_id(dependency_paths_id,10)

if __name__ == "__main__":
    main()