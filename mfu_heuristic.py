import numpy as np
from computeNcMat import computeNcMat
from buildFci import buildFci
from delayMat import delayMat


def mfu_heuristic(Rcpu, Rmem, Fcm, M, lambd, Rs, app_edge, delta_mes, RTT, Ne):
    ## VARIABLES INITIALIZATION ##
    e = 2
    Cost_cpu_edge = 1
    Cost_mem_edge = 1
    Ubit = np.arange(1, e+1) * M  # User position in the state vector
    Rcpu_req = np.tile(np.zeros(int(M)), e)  # Seconds of CPU per request (set to zero for all microservices)
    Rcpu_req[int(Ubit[0])-1] = 0   
    Rcpu_req[int(Ubit[1])-1] = 0
    Rs = np.append(Rs, 0)  # Add the user in the Rs vector
    Rs = np.tile(Rs, e)  # Expand the Rs vector to fit the number of data centers

    ## BUILD Sold VECTOR ##
    app_edge = np.append(app_edge, 1) # Add the user in app_edge vector (user is in the edge cluster)
    Sold_b = np.concatenate((np.ones(int(M)), app_edge))
    Sold_b[M-1] = 0  # User is not in the cloud
    
    ## COMPUTE THE COST (CPU + MEMORY) OF THE OLD STATE ##
    Sold_edge_b = Sold_b[M:2*M] # Binary placement status containing edge microservices only
    Rcpu_cloud = Rcpu[:M]
    Rmem_cloud = Rmem[:M]
    Rcpu_edge = Rcpu[M:2*M]
    Rmem_edge = Rmem[M:2*M]
    Rcpu_edge_old_sum = np.sum(Sold_edge_b * Rcpu_edge) # Total CPU requested by instances in the edge
    Rmem_edge_old_sum = np.sum(Sold_edge_b * Rmem_edge) # Total Memory requested by instances in the edge
    Cost_cpu_edge_old_sum = Cost_cpu_edge * Rcpu_edge_old_sum # Total CPU cost
    Cost_mem_edge_old_sum = Cost_mem_edge * Rmem_edge_old_sum # Total Mem cost
    Cost_edge_old = Cost_cpu_edge_old_sum + Cost_mem_edge_old_sum

    n_rounds = 0
    
    ## COMPUTE THE DELAY OF THE OLD STATE ##
    delay_old = delayMat(Sold_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2)
    
    delta_delay_new = 0
    Snew_b = Sold_b.copy()
    Snew_edge_b = app_edge.copy()
    ## OFFLOAD ##
    if delta_mes > 0:
        while delta_mes > delta_delay_new:

            n_rounds = n_rounds + 1
            Nc = computeNcMat(Fcm, M, 1)
            Nc_max=-1
            argmax = -1
            for i in range(M-1):
                 if Nc[i]>Nc_max and Snew_edge_b[i]==0:
                    argmax = i
                    Nc_max = Nc[i]

            #argmax = np.argmax(Nc_cloud)
            Snew_edge_b = Snew_b[M:2*M].copy()
            Snew_edge_b[argmax] = 1
            Snew_b = np.concatenate((np.ones(int(M)), Snew_edge_b))
            Snew_b[M-1] = 0
            

            ## COMPUTE THE NEW COST (CPU + MEMORY) OF THE NEW STATE ##
            Rcpu_edge_new_sum = np.sum(Snew_edge_b * Rcpu_edge) # CPU requested by the new state
            Rmem_edge_new_sum = np.sum(Snew_edge_b * Rmem_edge) # Total Memory requested by instances in the edge
            Cost_cpu_edge_new_sum = Cost_cpu_edge * Rcpu_edge_new_sum # Total CPU cost
            Cost_mem_edge_new_sum = Cost_mem_edge * Rmem_edge_new_sum # Total Mem cost
            Cost_edge_new = Cost_cpu_edge_new_sum + Cost_mem_edge_new_sum

            delta_delay_new = delay_old - delayMat(Snew_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay delta reached

            delta_cost_opt =  Cost_edge_new - Cost_edge_old # Cost variation
            
            if np.all(Snew_edge_b == 1):
                break
        
        ## UNOFFLOAD  ##
        else:
            while -delta_mes > delta_delay_new:
                if Snew_b is None:
                    Fci = np.matrix(buildFci(Sold_b, Fcm, M, e))
                else:
                    Fci = np.matrix(buildFci(Snew_b, Fcm, M, e))
                Nc = computeNcMat(Fci, M, e)
                Nc_cloud = Nc[:M-1] # Get the Nc values for the cloud microservices
                argmax = np.argmax(Nc_cloud)
                if Snew_b is None:
                    Snew_edge_b = app_edge.copy()
                else:
                    Snew_edge_b = Snew_b[M:2*M].copy()
                Snew_edge_b[argmax] = 1
                Snew_b = np.concatenate((np.ones(int(M)), Snew_edge_b))
                Snew_b[M-1] = 0
                

                ## COMPUTE THE NEW COST (CPU + MEMORY) OF THE NEW STATE ##
                Rcpu_edge_new_sum = np.sum(Snew_edge_b * Rcpu_edge) # CPU requested by the new state
                Rmem_edge_new_sum = np.sum(Snew_edge_b * Rmem_edge) # Total Memory requested by instances in the edge
                Cost_cpu_edge_new_sum = Cost_cpu_edge * Rcpu_edge_new_sum # Total CPU cost
                Cost_mem_edge_new_sum = Cost_mem_edge * Rmem_edge_new_sum # Total Mem cost
                Cost_edge_new = Cost_cpu_edge_new_sum + Cost_mem_edge_new_sum # Total cost

                delta_delay_new = delayMat(Snew_b, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) - delay_old # Delay delta reached

                delta_cost_opt = Cost_edge_old - Cost_edge_new  # Cost variation
                
                if np.all(Snew_edge_b == 1):
                    break
            

    path_id_n = [i for i, x in enumerate(Snew_edge_b) if x > 0]
    Fci_old = np.matrix(buildFci(Sold_b, Fcm, M, e))
    Nci_old = computeNcMat(Fci_old, M, e)
    Fci_new = np.matrix(buildFci(Snew_b, Fcm, M, e))
    Nci_new = computeNcMat(Fci_new, M, e)
    Rcpu_edge_new = Rcpu_edge.copy()
    Rcpu_cloud_new = Rcpu_cloud.copy()
    Rmem_edge_new = Rmem_edge.copy()
    Rmem_cloud_new = Rmem_cloud.copy()
    for k in path_id_n:
                if Nci_old[k]>0:
                    cloud_cpu_reduction = (1-Nci_new[k]/Nci_old[k]) * Rcpu_cloud[k]  # equal to edge cpu increase
                    cloud_mem_reduction = (1-Nci_new[k]/Nci_old[k]) * Rmem_cloud[k]  # equal to edge mem increase
                    Rcpu_edge_new[k] = Rcpu_edge_new[k] + cloud_cpu_reduction
                    Rmem_edge_new[k] = Rmem_edge_new[k] + cloud_mem_reduction
                    Rcpu_cloud_new[k] = Rcpu_cloud_new[k] - cloud_cpu_reduction
                    Rmem_cloud_new[k] = Rmem_edge_new[k] - cloud_mem_reduction

    Cost_edge_new = Cost_cpu_edge * np.sum(Rcpu_edge_new) + Cost_mem_edge * np.sum(Rmem_edge_new) # Total edge cost                

    return Snew_edge_b, Cost_edge_new, delta_delay_new, delta_cost_opt, n_rounds


        