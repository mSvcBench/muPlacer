import numpy as np
from computeNc import computeNc
from buildFci import buildFci
from computeDnTot import computeDnTot
from numpy import inf


def IA_heuristic(Rcpu_old, Rmem_old, Fcm, M, lambd, Rs, S_edge_old, delta_mes, RTT, Ne):
    Cost_cpu_edge = 1
    Cost_mem_edge = 1
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    Rcpu_edge_old_sum = np.sum(S_b_old[M:] * Rcpu_old[M:]) # Total CPU requested by instances in the edge
    Rmem_edge_old_sum = np.sum(S_b_old[M:] * Rmem_old[M:]) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_edge_old_sum # Total CPU cost
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_edge_old_sum # Total Mem cost
    Cost_edge_old = Cost_cpu_edge_old_sum + Cost_mem_edge_old_sum # Total cost of old state

    n_rounds = 0
    
    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))
    Nci_old = computeNc(Fci_old, M, 2)
    delay_old = computeDnTot(S_b_old, Nci_old, Fci_old, Rs, RTT, Ne, lambd, M)
    Nc = computeNc(Fcm, M, 1) 
    delta_delay_new = 0
    S_b_new = S_b_old.copy()
    n_rounds = 0

    # DEFINE DICTIONARY FOR INTERACTION AWARE MATRIX ##
    maxes = {
        "ms_i": 0,
        "ms_j": 0,
        "interaction_freq": -1
    }

    if delta_mes > 0:
        ## OFFLOAD ##
        while delta_mes > delta_delay_new:
            n_rounds = n_rounds + 1
            maxes["interaction_freq"] = -1
            maxes["ms_i"] = 0
            maxes["ms_j"] = 0

            ## FIND THE MICROSERVICES WITH THE MOST INTERACTIONS ##
            for i in range (M):
                for j in range (M):
                    if i==j:
                        continue
                    if S_b_new[i+M]==0 or S_b_new[j+M]==0:
                        x = Nc[i] * Fcm[i,j] +  Nc[j] * Fcm[j,i]
                        if x > maxes["interaction_freq"]:
                            maxes["interaction_freq"] = x
                            maxes["ms_i"] = i
                            maxes["ms_j"] = j

            S_b_new[maxes["ms_i"]+M] = 1
            S_b_new[maxes["ms_j"]+M] = 1
            
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delta_delay_new = delay_old - computeDnTot(S_b_new, Nci_new, Fci_new, Rs, RTT, Ne, lambd, M) 
            if np.all(S_b_new[M:] == 1):
                # all instances at the edge
                break
    
    ## UNOFFLOAD  ##
    else:
        print("ToDo")

    # compute final values
    Rcpu_new = np.zeros(2*M)
    Rmem_new = np.zeros(2*M)
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new = computeDnTot(S_b_new, Nci_new, Fci_new, Rs, RTT, Ne, lambd, M)
    delta_new = delay_old - delay_new
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
    delta_cost = Cost_edge_new - Cost_edge_old 

    return S_b_new[M:].astype(int), Cost_edge_new, delta_new, delta_cost, n_rounds


        
