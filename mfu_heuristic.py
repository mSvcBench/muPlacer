import numpy as np
from numpy import inf
from computeNc import computeNc
from buildFci import buildFci
from computeDTot import computeDTot


def mfu_heuristic(params):
    ## VARIABLES INITIALIZATION ##
        
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
    delay_old,_,_,_ = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)
    Nc = computeNc(Fcm, M, 1) 

    delay_decrease_new = 0
    S_b_new = S_b_old.copy()
    
    ## OFFLOAD ##
    if delay_decrease_target > 0:
        while delay_decrease_target > delay_decrease_new:
            n_rounds = n_rounds + 1
            Nc_max=-1
            argmax = -1
            for i in range(M-1):
                 if Nc[i]>Nc_max and S_b_new[i+M]==0:
                    argmax = i
                    Nc_max = Nc[i]
            S_b_new[argmax+M] = 1
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delay_new,_,_,_ = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M) 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                break
        
    ## UNOFFLOAD  ##
    else:
        while -delay_decrease_target > delay_decrease_new:
            n_rounds = n_rounds + 1
            Nc_min=inf
            argmin = -1
            for i in range(M-1):
                if Nc[i]<Nc_min and S_b_new[i+M]==1:
                    argmin = i
                    Nc_min = Nc[i]
            S_b_new[argmin+M] = 0
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delay_new,_,_,_ = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M) 
            delay_decrease_new = delay_old - delay_new  
            if delay_decrease_new > -delay_decrease_target:
                    # excessive delay increase, revert and exit
                    S_b_new[argmin+M] = 1
            if np.all(S_b_new[M:2*M-1] == 0):
                # no other instance at the edge
                break
                
                
    # compute final values
   # compute final values
    Rcpu_new = np.zeros(2*M)
    Rmem_new = np.zeros(2*M)
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


        
