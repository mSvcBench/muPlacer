# pylint: disable=C0103, C0301

import datetime
import numpy as np
import networkx as nx
from computeNc import computeNcMat
from buildFci import buildFci
from buildFci import buildFcinew
from S2id import S2id
from old.delayMat import delayMat 
from old.delayMat import delayMatNcFci
from id2S import id2S
from numpy import inf

np.seterr(divide='ignore', invalid='ignore')


def offload_fast(Rcpu_curr, Rmem_curr, Fcm, M, lambd, Rs, S_edge_b_curr, delta_mes, RTT, Ne, depth):
    #x = datetime.datetime.now().strftime('%d-%m_%H:%M:%S')
    #filename = f'offload_{x}.mat'
    #np.save(filename, arr=[Rcpu, Rmem, Fcm_nocache, M, lambd, Rs, app_edge, min_delay_delta, RTT])


    ## INITIALIZE VARIABLES ##
    S_edge_b_curr = np.append(S_edge_b_curr, 1) # Add the user in app_edge vector (user is in the edge cluster)
    S_b_curr = np.concatenate((np.ones(int(M)), S_edge_b_curr))
    S_b_curr[M-1] = 0  # User is not in the cloud
    Ce = np.inf # CPU capacity of edge datacenter
    Me = np.inf # Memory capacity of edge datacenter
    Rs = np.append(Rs, 0)  # Add the user in the Rs vector
    Rs = np.tile(Rs, 2)  # Expand the Rs vector to fit the number of data centers
    cost_cpu_edge = 1
    cost_mem_edge = 1
    delta_target = delta_mes
    
    # Seconds of CPU per request (set to zero for all microservices since edge/cloud computing delay is the same)
    Rcpu_req = np.tile(np.zeros(int(M)), 2)  
    Rcpu_req[M-1] = 0   
    Rcpu_req[2*M-1] = 0

    
    Fci = np.matrix(buildFcinew(S_b_curr, Fcm, M)) # instance-set call frequency matrix
    Nci = computeNcMat(Fci, M, 2)  # number of request per instance per user request
    
    delay_curr = delayMatNcFci(S_b_curr, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci, Fci,2)  # current delay
  
    
    Rcpu_edge_curr_sum = np.sum(S_b_curr[M:2*M] * Rcpu_curr[M:]) # Total CPU requested by instances in the edge
    Rmem_edge_curr_sum = np.sum(S_b_curr[M:2*M] * Rmem_curr[M:]) # Total Memory requested by instances in the edge
    cost_cpu_edge_curr_sum = cost_cpu_edge * Rcpu_edge_curr_sum # Total CPU cost
    cost_mem_edge_curr_sum = cost_mem_edge * Rmem_edge_curr_sum # Total Mem cost
    cost_edge_curr = cost_cpu_edge_curr_sum + cost_mem_edge_curr_sum # Total cost

    
    S_b_opt = S_b_curr  # Inizialize the new edge status
    S_b_temp = np.empty_like(S_b_curr)
    S_b_new = np.empty_like(S_b_curr)
    S_b_temp = np.empty_like(S_b_new)
    Rcpu_opt = np.empty_like(Rcpu_curr)
    Rmem_opt = np.empty_like(Rmem_curr)
    Rcpu_new = np.empty_like(Rcpu_curr)
    Rmem_new = np.empty_like(Rmem_curr)
    Rcpu_temp = np.empty_like(Rcpu_curr)
    Rmem_temp = np.empty_like(Rmem_curr)
    
    np.copyto(Rcpu_opt,Rcpu_curr)
    np.copyto(Rmem_opt,Rmem_curr)
    delay_opt = delay_curr

    debug = False
    debug2 = False
    skip_neg = False
    delta_cost_opt=0
    delta_opt=1
    

    while True:
        print(f'-----------------------') if debug else 0
        w_min = float("inf") # Initialize the weight
        skip_neg = False
        np.copyto(Rcpu_new,Rcpu_opt)
        np.copyto(Rmem_new,Rmem_opt)
        np.copyto(S_b_new, S_b_opt) # New edge status in binary encoding
        edge_ids = np.argwhere(S_b_new[M:]>0) # ids of microservices at the edge
        u = Fcm * S_b_new[M:].reshape(M,1).repeat(M,axis=1) # Fcm matrix with values for callers that are at the eddge. Colums with values different from 0 means that the microservice related to the column ha an upstream in the edge and therefroe is a candidate  
        candidate_ms = np.argwhere(np.sum(u,axis=0))
        candidate_ms = [i for i in candidate_ms if i not in edge_ids] # remove from candidate microservices those that are already at edge

        delay_new = delay_opt # Delay of the new placement state
        cost_edge_new  = cost_cpu_edge * np.sum(Rcpu_new[M:]) + cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
        print(f'new edge micros {np.argwhere(S_b_new[M:]==1).squeeze()}, delta_delay {1000*(delay_curr-delay_new)}, cost {cost_edge_new}, delta_cost/delta_delay {delta_cost_opt/(1000*delta_opt)}') if debug else 0
        
        # Check if the delay reduction is enough
        if delay_curr-delay_new >= delta_target:
            #delay reduction reached
            break
        # compute candidate microservices, those with at leas a parent @ edge

        if len(candidate_ms) == 0:
            # All dependency path considered no other way to reduce delay
            break
        for ms_id in candidate_ms :
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[M+ms_id] = 1
            Fci_temp = np.matrix(buildFcinew(S_b_temp, Fcm, M))
            Nci_temp = computeNcMat(Fci_temp, M, 2)
            delay_temp = delayMatNcFci(S_b_temp, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_temp, Fci_temp, 2) # Delay of the new placement state
            delta_delay = delay_new - delay_temp
            if skip_neg and delta_delay<0:
                continue
            
            # compute the cost increase adding this microservice
            np.copyto(Rcpu_temp,Rcpu_curr) 
            np.copyto(Rmem_temp,Rmem_curr) 
            cloud_cpu_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rcpu_curr[:M]  # reduction of cloud CPU request 
            cloud_mem_reduction = (1-Nci_temp[:M]/Nci[:M]) * Rmem_curr[:M]  # reduction of cloud mem request 
            cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
            cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
            cloud_cpu_reduction[cloud_cpu_reduction==-inf] = 0
            cloud_mem_reduction[cloud_mem_reduction==-inf] = 0
            Rcpu_temp[M:] = Rcpu_temp[M:] + cloud_cpu_reduction
            Rmem_temp[M:] = Rmem_temp[M:] + cloud_mem_reduction
            Rcpu_temp[:M] = Rcpu_temp[:M] - cloud_cpu_reduction
            Rmem_temp[:M] = Rmem_temp[:M] - cloud_mem_reduction
            cost_edge_temp = cost_cpu_edge * np.sum(Rcpu_temp[M:]) + cost_mem_edge * np.sum(Rmem_temp[M:]) # Total edge cost
            delta_cost = cost_edge_temp - cost_edge_new
            
            r_delta = delta_target - (delay_curr-delay_new) # residul delay to decrease wrt previous conf
            if delta_delay < 0:
                w = 1e6 - delta_cost * (1000*delta_delay) 
                #w = 1e6 - (delta_cost/delay_curr) / max((delta_target-delta_delay)/delta_target, 1e-3)
            else:
                w = delta_cost /  min(1000*delta_delay, 1000*r_delta)
                skip_neg = True
            
            print(f'new edge micro {ms_id}, cost {delta_cost}, delta_delay {1000*delta_delay}, weight {w}') if debug2 else 0

            if w < w_min:
                np.copyto(S_b_opt, S_b_temp)
                np.copyto(Rcpu_opt,Rcpu_temp)
                np.copyto(Rmem_opt,Rmem_temp)
                delta_cost_opt = delta_cost
                delta_opt = delta_delay
                delay_opt = delay_temp
                w_min = w
            
    # cleaning phase
    while True:
        c_max=0
        i_max=-1
        for i in range(M, 2*M-1):
            if S_b_new[i]==1:
                # try remove microservice
                np.copyto(S_b_temp,S_b_new)
                S_b_temp[i] = 0
                delta_final2 = delay_curr - delayMat(S_b_temp, Fcm, Rcpu_new, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # delay delta reached
                if delta_final2>=delta_target:
                    # possible removal
                    if Rcpu_new[i]*cost_cpu_edge > c_max:
                        i_max = i
                        c_max = Rcpu_new[i]*cost_cpu_edge
        if i_max>-1:
            print('cleaning')
            S_b_new[i_max] = 0
            cost_edge_new = cost_edge_new - Rcpu_new[i_max]*cost_cpu_edge
        else:
            break
            
    n_rounds = 1
    # compute final values
    Fci_new = np.matrix(buildFcinew(S_b_new, Fcm, M))
    Nci_new = computeNcMat(Fci_new, M, 2)
    delay_new = delayMatNcFci(S_b_new, Fcm, Rcpu_curr, Rcpu_req, RTT, Ne, lambd, Rs, M, Nci_new, Fci_new, 2) # Delay of the new placement state
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
    Cost_edge_new = cost_cpu_edge * np.sum(Rcpu_new[M:]) + cost_mem_edge * np.sum(Rmem_new[M:]) # Total edge cost
    delta_cost = Cost_edge_new - cost_edge_curr 
    
    return S_b_new[M:].astype(int).tolist(), Cost_edge_new, delta_new, delta_cost, n_rounds