# pylint: disable=C0103, C0301
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import networkx as nx
from computeNc import computeNc
from buildFci import buildFci
from numpy import inf
from computeDTot import computeDTot
import logging
import sys
import utils
from S2id import S2id
from id2S import id2S
import time
#from EPAMP_unoffload5 import unoffload


np.seterr(divide='ignore', invalid='ignore')

# Set up logger
logger = logging.getLogger('EPAMP_offload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP offload %(levelname)s %(message)s'))
logger.propagate = False

def offload(params):

    ## INITIALIZE VARIABLES ##
    # S_edge_old (M,) vector of binary values indicating if the microservice instance-set is running in the edge or not
    # Acpu_old (2*M,) vector of actual CPU req by instance-set at the cloud (:M) and at the edge (M:)
    # Amem_old (2*M,) vector of actual Memory req by instance-set at the cloud (:M) and at the edge (M:)
    # Fcm(M,M) microservice call frequency matrix
    # M number of microservices
    # lambd user request rate
    # Rs(M,) vector of response size of microservices
    # Di(M,) vector of internal delay of microservices
    # delay_decrease_target delay reduction target
    # RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    # Ne cloud-edge network bitrate
    # Cost_cpu_edge cost of CPU unit at the edge
    # Cost_mem_edge cost of Memory unit at the edge
    # Cost_cpu_cloud cost of CPU unit at the cloud
    # Cost_mem_cloud cost of Memory unit at the cloud
    # u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)
    # Qmem (M,) memory quantum in bytes, Kubernetes memory request
    # Qcpu (M,) CPU quantum in cpu sec, Kubernetes CPU request
    # look_ahead look ahead factor to increase the delay decrease target
    # dependency_paths_b (N,M) binary-based (b) encoded dependency paths precomputed
    # locked_b (M,) binary encoding of microservice that can not be moved at the edge
    # sweeping_limit maximum number of gateway childs added in the sweeping building phase

    def cache_probe(S_b, round):
        result=dict()
        result['delay'] = None
        result['Acpu'] = None
        result['Amem'] = None
        result['Fci'] = None
        result['Nci'] = None
        result['rhoce'] = None
        result['cost'] = None
        hit = False
        global_cache['cache_access'] += 1
        S_id_edge=np.array2string(S_b[global_M:])
        if S_id_edge in global_cache['delay']:
            logger.debug(f'cache_hit for {np.argwhere(S_b[global_M:]==1).squeeze()}')
            global_cache['cache_hit'] += 1
            result['delay'] = global_cache['delay'][S_id_edge]
            result['Acpu'] = np.copy(global_cache['Acpu'][S_id_edge])
            result['Amem'] = np.copy(global_cache['Amem'][S_id_edge])
            result['Fci'] = global_cache['Fci'][S_id_edge].copy()
            result['Nci'] = global_cache['Nci'][S_id_edge].copy()
            result['rhoce'] = global_cache['rhoce'][S_id_edge]
            result['cost'] = global_cache['cost'][S_id_edge]
            global_cache['expire'][S_id_edge] = round
            hit = True
        return hit, result
    
    def evaluate_perf(S_b_new, Acpu_old, Amem_old, Nci_old, round):
        hit, result = cache_probe(S_b_new, round)
        if not hit:
            Acpu_new = np.zeros(2*global_M)
            Amem_new = np.zeros(2*global_M)
            Fci_new = np.matrix(buildFci(S_b_new, global_Fcm, global_M))
            Nci_new = computeNc(Fci_new, global_M, 2)
            delay_new,_,_,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))
            utils.computeResourceShift(Acpu_new, Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old)
            cost_new = utils.computeCost(Acpu_new, Amem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new*global_Ne, global_Cost_network)[0] # Total  cost of the temp state
            # cache insertion
            cache_insert(S_b_new, delay_new, Acpu_new, Amem_new, Fci_new, Nci_new, rhoce_new, cost_new, round)
            result = dict()
            result['delay'] = delay_new
            result['Acpu'] = Acpu_new.copy()
            result['Amem'] = Amem_new.copy()
            result['Fci'] = Fci_new.copy()
            result['Nci'] = Nci_new.copy()
            result['rhoce'] = rhoce_new
            result['cost'] = cost_new
        return hit, result
    
    # def evaluate_perf_no_caching(S_b_new, Acpu_old, Amem_old, Nci_old, round):
    #     # fake caching test
    #     Acpu_new_p = np.zeros(2*global_M)
    #     Amem_new_p = np.zeros(2*global_M)
    #     Fci_new_p = np.matrix(buildFci(S_b_new, global_Fcm, global_M))
    #     Nci_new_p = computeNc(Fci_new_p, global_M, 2)
    #     delay_new_p,_,_,rhoce_new_p = computeDTot(S_b_new, Nci_new_p, Fci_new_p, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))
    #     utils.computeResourceShift(Acpu_new_p, Amem_new_p,Nci_new_p,Acpu_old,Amem_old,Nci_old)
    #     cost_new_p = utils.computeCost(Acpu_new_p, Amem_new_p, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new_p*global_Ne, global_Cost_network)[0] # Total  cost of the temp state
    #     result = dict()
    #     result['delay'] = delay_new_p
    #     result['Acpu'] = Acpu_new_p.copy()
    #     result['Amem'] = Amem_new_p.copy()
    #     result['Fci'] = Fci_new_p.copy()
    #     result['Nci'] = Nci_new_p.copy()
    #     result['rhoce'] = rhoce_new_p
    #     result['cost'] = cost_new_p
    #     global_cache['cache_access'] += 1
    #     hit = False
    #     return hit, result
    
    def cache_insert(S_b, delay, Acpu, Amem, Fci, Nci, rhoce, cost, round):
        S_id_edge=np.array2string(S_b[global_M:])
        global_cache['delay'][S_id_edge] = delay
        global_cache['rhoce'][S_id_edge] = rhoce
        global_cache['Acpu'][S_id_edge]=np.copy(Acpu)
        global_cache['Amem'][S_id_edge]=np.copy(Amem)
        global_cache['expire'][S_id_edge] = round
        global_cache['Fci'][S_id_edge]=Fci.copy()
        global_cache['Nci'][S_id_edge]=Nci.copy()
        global_cache['cost'][S_id_edge]=cost
        logger.debug(f'cache insert for {np.argwhere(S_b_temp[global_M:]==1).squeeze()}')
    
    def cache_cleaning(round):
        for key in list(global_cache['delay'].keys()):
            if global_cache['expire'][key] + global_cache_ttl < round:
                del global_cache['delay'][key]
                del global_cache['rhoce'][key]
                del global_cache['Acpu'][key]
                del global_cache['Amem'][key]
                del global_cache['expire'][key]
    
    def dp_builder_with_single_path_adding(S_b_init, Acpu_init, Amem_init, Nci_init, round):
        ## BUILDING OF COMPOSITE DEPENDENCY PATHS WITH SINGLE PATH ADDING##
        
        nonlocal dependency_paths_b_full_built, dependency_paths_b_full
        if not dependency_paths_b_full_built:
            for ms in range(global_M-1):
                paths_n = list(nx.all_simple_paths(global_G, source=global_M-1, target=ms)) 
                for path_n in paths_n:
                    # path_n numerical id (n) of the microservices of the dependency path
                    # If all microservices in the path are in the edge this path is not a cloud-joined path
                    if all(global_S_b_old[global_M+np.array([path_n])].squeeze()==1):
                        continue
                    else:
                        path_b = np.zeros((1,global_M),int)
                        path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                        dependency_paths_b_full = np.append(dependency_paths_b_full,path_b,axis=0)
            dependency_paths_b_full_built = True
        residual = np.sum(np.maximum(dependency_paths_b_full-S_b_init[global_M:],0),axis=1)
        rl = np.argwhere((residual > 0) & (residual <= global_u_limit)).flatten()
        return dependency_paths_b_full[rl]
    
    def dp_builder_traces(S_b_init, Acpu_init, Amem_init, Nci_init, round):
        ## BUILDING OF SIMULATION TRACES#
        nonlocal dependency_paths_b_full_built, dependency_paths_b_full
        hit, result = evaluate_perf(S_b_init, Acpu_init, Amem_init, Nci_init, round)
        delay_init = result['delay']
        Fci_init = result['Fci']
        cost_init = result['cost']
        if not dependency_paths_b_full_built:
            n_traces = global_max_traces
            dependency_paths_b_full = np.empty((0,global_M), int)
            user = global_M-1
            iteration = 0
            while True:
                iteration += 1
                trace_sample_b = np.zeros(global_M)
                trace_sample_b = dp_builder_trace(user,trace_sample_b,global_Fcm, S_b_init)
                dependency_paths_b_full = np.append(dependency_paths_b_full, trace_sample_b.reshape(1, -1), axis=0)
                # if not any(np.array_equal(trace_sample_b, row) for row in dependency_paths_b_full):
                #    dependency_paths_b_full = np.append(dependency_paths_b_full, trace_sample_b.reshape(1, -1), axis=0)
                if len(dependency_paths_b_full) >= n_traces or (iteration > 100*n_traces and len(dependency_paths_b_full) > 20):
                    break
            trace_sample_b = np.ones(global_M)  # add full edge trace
            dependency_paths_b_full = np.append(dependency_paths_b_full, trace_sample_b.reshape(1, -1), axis=0)
            dependency_paths_b_full_built = True
        
        dependency_paths_b = np.empty((0,global_M), int)
        # remove traces fully in the edge
        residual = np.argwhere(np.sum(np.maximum(dependency_paths_b_full-S_b_init[global_M:],0),axis=1)>0).flatten()
        dependency_paths_b = dependency_paths_b_full[residual]
        
        # clean from these traces the cloud microservices that are at a distance greather than u_limit from the edge gateways
        edge_gws = np.unique(np.argwhere(Fci_init[global_M:2*global_M,0:global_M]>0)[:,0]) # list of edge gateways: microservices in the edge with at least one call from the cloud
        allowed_cloud_ms = np.empty((0), int)
        for edge_gw in edge_gws:
            allowed_cloud_ms = np.append(allowed_cloud_ms, np.argwhere(global_ms_distances[edge_gw][:] <= global_u_limit).flatten())
        allowed_cloud_ms = np.unique(allowed_cloud_ms)
        not_allowed_ms = np.setdiff1d(np.arange(global_M), allowed_cloud_ms)
        dependency_paths_b[:,not_allowed_ms]=0
        
        # compute the frequency of the dependency paths to return the most frequently used
        dependency_paths_b, paths_freq = np.unique(dependency_paths_b, axis=0,return_counts=True)
        mfu_dependency_paths_id = np.flip(np.argsort(paths_freq))

        return dependency_paths_b[mfu_dependency_paths_id[:min(global_max_dps,len(mfu_dependency_paths_id))]]

    
    def dp_builder_trace(node,trace,global_Fcm, S_b_init):
        children = np.argwhere(global_Fcm[node,0:global_M]>0).flatten()
        for child in children:
            if np.random.random() < global_Fcm[node,child]:
                trace[child] = 1
                trace = dp_builder_trace(child,trace,global_Fcm,S_b_init)
        return trace

    def dp_builder_with_minimum_sweeping(S_b_init, Acpu_init, Amem_init, Nci_init, round):
        ## BUILDING OF COMPOSITE DEPENDENCY PATHS WITH MINIMUM SWEEPING##
        
        dependency_paths_b = np.empty((0,global_M), int) # Storage of binary-based (b) encoded dependency paths
        hit, result = evaluate_perf(S_b_init, Acpu_init, Amem_init, Nci_init, round)
        delay_init = result['delay']
        Fci_init = result['Fci']
        cost_init = result['cost']

        S_b_sweeping_temp = np.copy(S_b_init) # S_b_sweeping_temp is the temporary placement state used in bulding dependency paths with sweeping

        cloud_gws = np.unique(np.argwhere(Fci_init[global_M:2*global_M,0:global_M]>0)[:,0]) # list of cloud gateways: microservices in the cloud with at least one call from the edge
        cloud_gws = cloud_gws[np.argwhere(global_locked_b[cloud_gws]==0)] # remove cloud gateways that can not be moved @ the edge
        for cgw in cloud_gws:
            d = 1
            delay_sweeping_opt = delay_init
            delay_sweeping_new = delay_init
            cost_sweeping_new = cost_init
            path_b_sweep_new = np.zeros((1,global_M),int)  # composite dependency path built during the sweeping iteration 
            path_b_sweep_temp = np.zeros((1,global_M),int)
            while True:
                cloud_gw_children = np.argwhere(global_ms_distances[cgw][0]==d).flatten() # list of microservices called by the cloud gateway
                cloud_gw_children=cloud_gw_children[np.argwhere(global_locked_b[cloud_gw_children]==0)] # remove children that can not be moved @ the edge
                if len(cloud_gw_children)==0:
                    logger.warning(f"sweeping for cloud gateway {cgw} didn't find any suitable subgraph to add for latency reduction")
                    break
                while len(cloud_gw_children)>0:
                    w_min_sweeping = float("inf") # Initialize the minimum weight
                    np.copyto(path_b_sweep_temp,path_b_sweep_new)
                    for ch in cloud_gw_children:
                        S_b_sweeping_temp[global_M+ch] = 1
                        # cache access
                        _, result = evaluate_perf(S_b_sweeping_temp, Acpu_init, Amem_init, Nci_init, round) 
                        delay_sweeping_temp = result['delay']
                        rhoce_sweeping_temp = result['rhoce']
                        cost_sweeping_temp = result['cost']
                        # r_delay_sweeping_decrease = global_delay_decrease_target * global_look_ahead - (global_delay_old-delay_sweeping_new) # residul delay to decrease wrt previous sweep
                        r_delay_sweeping_decrease = 1e6
                        delay_sweeping_decrease_temp = delay_sweeping_new - delay_sweeping_temp
                        cost_sweeping_increase_temp = cost_sweeping_temp - cost_sweeping_new
                        if delay_sweeping_decrease_temp <= 0:  
                            wi = 1e6 + cost_sweeping_increase_temp *  1000 * abs(delay_sweeping_decrease_temp)
                        else:
                            wi = cost_sweeping_increase_temp /  max(min(1000*delay_sweeping_decrease_temp, 1000*r_delay_sweeping_decrease),1e-3) # 1e-3 used to avoid division by zero                
                        S_b_sweeping_temp[global_M+ch] = 0
                        if wi < w_min_sweeping:
                            w_min_sweeping = wi
                            ch_sweeping_opt = ch
                            delay_sweeping_opt = delay_sweeping_temp
                            cost_sweeping_opt = cost_sweeping_temp
                            rhoce_sweeping_opt = rhoce_sweeping_temp
                        
                    path_b_sweep_temp[0,ch_sweeping_opt] = 1 
                    S_b_sweeping_temp[global_M+ch_sweeping_opt] = 1
                    if delay_sweeping_opt < delay_init:
                        # minimum delay reduction obtained move to next gateway
                        dependency_paths_b = np.append(dependency_paths_b,path_b_sweep_temp,axis=0)
                        logger.info(f'builder included dependency path {np.argwhere(path_b_sweep_temp[0]==1).flatten()}, cost increase {cost_sweeping_opt-cost_init}, delay decrease {1000*(delay_init-delay_sweeping_opt)} ms, delay {delay_sweeping_opt} ms')
                        break
                    np.copyto(path_b_sweep_new,path_b_sweep_temp)
                    delay_sweeping_new = delay_sweeping_opt
                    cost_sweeping_new = cost_sweeping_opt
                    cloud_gw_children = np.delete(cloud_gw_children,np.argwhere(cloud_gw_children==ch_sweeping_opt)[0,0])
                if delay_sweeping_opt < delay_init:
                    break
                else:
                    d += 1
                
        return dependency_paths_b

    # def dp_builder_with_sweeping(Fcm, M, S_b_old, Di, Rs, RTT, Ne, lambd, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, Cost_network, Qcpu, Qmem, delay_decrease_target, look_ahead, cache, locked_b, sweeping_limit):
    #     ## BUILDING OF COMPOSITE DEPENDENCY PATHS WITH SWEEPING##
    #     dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
    #     S_b_sweeping_temp = np.zeros(2*M) # S_b_sweeping_temp is the temporary placement state used in bulding dependency paths with sweeping
    #     Acpu_sweeping_temp = np.zeros(2*M)   # Acpu_temp is the temporary CPU request vector used in bulding dependency paths
    #     Amem_sweeping_temp = np.zeros(2*M)   # Amem_temp is the temporary Memory request vector used in bulding dependency paths
    #     cloud_gws = np.unique(np.argwhere(Fci_new[M:2*M,0:M]>0)[:,1]) # list of cloud gateways: microservices in the cloud with at least one call from the edge
    #     cloud_gws = cloud_gws[np.argwhere(locked_b[cloud_gws]==0)] # remove cloud gateways that can not be moved @ the edge
    #     for cgw in cloud_gws:
    #         cloud_gw_children = np.argwhere(np.ravel(Fci_new[cgw,0:M])>0).flatten() # list of microservices called by the cloud gateway
    #         cloud_gw_children=cloud_gw_children[np.argwhere(locked_b[cloud_gw_children]==0)] # remove children that can not be moved @ the edge
    #         path_b_gw = np.zeros((1,M),int)
    #         path_b_gw[0,:] = S_b_new[M:].copy()
    #         path_b_gw[0,cgw] = 1 # dependency path with the cloud gateway only
    #         dependency_paths_b = np.append(dependency_paths_b,path_b_gw,axis=0)
    #         np.copyto(S_b_sweeping_temp,S_b_new)
    #         S_b_sweeping_temp[M+cgw] = 1
            
    #         # cache probing
    #         hit, result = cache_probe(S_b_sweeping_temp, round)
    #         if hit:
    #             delay_sweeping_temp = result['delay']
    #             Acpu_sweeping_temp = result['Acpu']
    #             Amem_sweeping_temp = result['Amem']
    #             Fci_sweeping_temp = result['Fci']
    #             rhoce_sweeping_temp = result['rhoce']
    #             cost_sweeping_temp = result['cost']
    #         else:
    #             Fci_sweeping_temp = np.matrix(buildFci(S_b_sweeping_temp, Fcm, M))
    #             Nci_sweeping_temp = computeNc(Fci_sweeping_temp, M, 2)
    #             delay_sweeping_temp,_,_,rhoce_sweeping_temp = computeDTot(S_b_sweeping_temp, Nci_sweeping_temp, Fci_sweeping_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))
    #             utils.computeResourceShift(Acpu_sweeping_temp, Amem_sweeping_temp,Nci_sweeping_temp,global_Acpu_old,global_Amem_old,global_Nci_old)
    #             cost_sweeping_temp = utils.computeCost(Acpu_sweeping_temp, Amem_sweeping_temp, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_sweeping_temp*Ne, Cost_network)[0] # Total  cost of the temp state
    #             # cache insertion
    #             cache_insert(S_b_sweeping_temp, delay_sweeping_temp, Acpu_sweeping_temp, Amem_sweeping_temp, Fci_sweeping_temp, rhoce_sweeping_temp, cost_sweeping_temp, round, cache)
            
              
    #         # iterative children sweeping
    #         delay_sweeping_new = delay_sweeping_temp
    #         cost_sweeping_new = cost_sweeping_temp
    #         path_b_sweep_new = path_b_gw.copy()
    #         path_b_sweep_temp = path_b_gw.copy()
    #         added_children = 0
    #         while len(cloud_gw_children)>0:
    #             if added_children >= sweeping_limit:
    #                 logger.info(f"reached sweeping limit")
    #                 break
    #             w_min_sweeping = float("inf") # Initialize the minimum weight
    #             np.copyto(path_b_sweep_temp,path_b_sweep_new)
    #             for ch in cloud_gw_children:
    #                 S_b_sweeping_temp[M+ch] = 1
                    
    #                 # cache probing
    #                 hit, result = cache_probe(S_b_sweeping_temp, round, cache) 
    #                 if hit:
    #                     delay_sweeping_temp = result['delay']
    #                     Acpu_sweeping_temp = result['Acpu']
    #                     Amem_sweeping_temp = result['Amem']
    #                     Fci_sweeping_temp = result['Fci']
    #                     rhoce_sweeping_temp = result['rhoce']
    #                     cost_sweeping_temp = result['cost']
    #                 else:
    #                     Fci_sweeping_temp = np.matrix(buildFci(S_b_sweeping_temp, Fcm, M))
    #                     Nci_sweeping_temp = computeNc(Fci_sweeping_temp, M, 2)
    #                     delay_sweeping_temp,_,_,rhoce_sweeping_temp = computeDTot(S_b_sweeping_temp, Nci_sweeping_temp, Fci_sweeping_temp, Di, Rs, RTT, Ne, lambd, M, np.empty(0))
    #                     utils.computeResourceShift(Acpu_sweeping_temp,Amem_sweeping_temp,Nci_sweeping_temp,global_Acpu_old,global_Amem_old,global_Nci_old)
    #                     cost_sweeping_temp = utils.computeCost(Acpu_sweeping_temp, Amem_sweeping_temp, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_sweeping_temp*Ne, Cost_network)[0]# Total cost of the temp state
    #                     # cache insertion
    #                     cache_insert(S_b_sweeping_temp, delay_sweeping_temp, Acpu_sweeping_temp, Amem_sweeping_temp, Fci_sweeping_temp, rhoce_sweeping_temp, cost_sweeping_temp, round, cache)
                    
    #                 r_delay_sweeping_decrease = delay_decrease_target * look_ahead - (global_delay_old-delay_sweeping_new) # residul delay to decrease wrt previous sweep
    #                 delay_sweeping_decrease_temp = delay_sweeping_new - delay_sweeping_temp
    #                 cost_sweeping_increase_temp = cost_sweeping_temp - cost_sweeping_new
    #                 if rhoce_sweeping_temp == 1 or delay_sweeping_decrease_temp <= 0:  
    #                     wi = 1e6 - cost_sweeping_increase_temp *  1000 * delay_sweeping_decrease_temp
    #                 else:
    #                     wi = cost_sweeping_increase_temp /  max(min(1000*delay_sweeping_decrease_temp, 1000*r_delay_sweeping_decrease),1e-3) # 1e-3 used to avoid division by zero                
    #                 S_b_sweeping_temp[M+ch] = 0
    #                 if wi < w_min_sweeping:
    #                     w_min_sweeping = wi
    #                     ch_sweeping_best = ch
    #                     delay_sweeping_opt = delay_sweeping_temp
    #                     cost_sweeping_opt = cost_sweeping_temp
                    
    #             path_b_sweep_temp[0,ch_sweeping_best] = 1 
    #             S_b_sweeping_temp[M+ch_sweeping_best] = 1
    #             dependency_paths_b = np.append(dependency_paths_b,path_b_sweep_temp,axis=0)
    #             np.copyto(path_b_sweep_new,path_b_sweep_temp)
    #             delay_sweeping_new = delay_sweeping_opt
    #             cost_sweeping_new = cost_sweeping_opt
    #             cloud_gw_children = np.delete(cloud_gw_children,np.argwhere(cloud_gw_children==ch_sweeping_best)[0,0])
    #             added_children += 1
    #             if global_delay_old-delay_sweeping_new >= delay_decrease_target * look_ahead:
    #                 # delay reduction reached no need to proceed sweeping with this gw
    #                 break
    #     return dependency_paths_b

    # def dp_builder_with_random_composite_path_adding(Fcm, M, S_b_old, Di, Rs, RTT, Ne, lambd, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, Cost_network, Qcpu, Qmem, delay_decrease_target, look_ahead, cache, locked_b, sweeping_limit):
    #     ## BUILDING OF COMPOSITE DEPENDENCY PATHS BY RANDOM COMBINATION OF SINGLE DP##
    #     nonlocal dependency_paths_b_full_built, dependency_paths_b_full
    #     if not dependency_paths_b_full_built:
    #         for ms in range(M-1):
    #             paths_n = list(nx.all_simple_paths(global_G, source=M-1, target=ms)) 
    #             for path_n in paths_n:
    #                 # path_n numerical id (n) of the microservices of the dependency path
    #                 # If all microservices in the path are in the edge this path is not a cloud-joined path
    #                 if all(S_b_old[M+np.array([path_n])].squeeze()==1):
    #                     continue
    #                 else:
    #                     path_b = np.zeros((1,M),int)
    #                     path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
    #                     dependency_paths_b_full = np.append(dependency_paths_b_full,path_b,axis=0)
    #         dependency_paths_b_full_built = True
        
    #     residual = np.sum(np.maximum(dependency_paths_b_full-S_b_new[M:],0),axis=1)
    #     rl1 = np.argwhere(residual == 1).flatten() # dependency paths that include a cloud-gateway only
    #     dependency_paths_b = np.empty((0,M), int)
    #     S_b_temp = np.zeros(2*M)
    #     for i in rl1:
    #         path_b = dependency_paths_b_full[i].reshape(1,M)
    #         dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0) # add cloud-gateway only deppendency paths
    #         S_b_temp[M:] = np.maximum(S_b_new[M:] + path_b,1)
    #         residual = np.sum(np.maximum(dependency_paths_b_full-S_b_temp[M:],0),axis=1)
    #         rl2 = np.argwhere(residual == 1).flatten() # dependency paths that include a cloud-gateway and a child
    #         n_combs = 2**len(rl2)
    #         # Generate n_combs unique integer numbers without repetition
    #         unique_combinations = np.random.choice(n_combs, size=min(n_combs,128), replace=False)
    #         # Generate random composite dependency paths
    #         for i in unique_combinations:
    #             path_b = np.zeros((1,M),int)
    #             binary_str = np.binary_repr(i, width=len(rl2))
    #             dp_indexes = np.argwhere(np.array(list(binary_str))=='1').flatten()
    #             if len(dp_indexes)==0:
    #                 continue
    #             path_b = np.sum(dependency_paths_b_full[rl2[dp_indexes]],axis=0)
    #             path_b = np.minimum(path_b,1)
    #             dependency_paths_b = np.append(dependency_paths_b,path_b.reshape(1,M),axis=0)
    #     return dependency_paths_b


    
    # mandatory paramenters
    global_S_edge_old = params['S_edge_b']
    global_Acpu_old = params['Acpu']
    global_Amem_old = params['Amem']
    global_Fcm = params['Fcm']
    global_M = params['M']
    global_lambd = params['lambd']
    global_Rs = params['Rs']
    global_delay_decrease_target = params['delay_decrease_target']
    global_RTT = params['RTT']
    global_Ne = params['Ne']
    global_Cost_cpu_edge = params['Cost_cpu_edge'] # Cost of CPU unit at the edge per hours
    global_Cost_mem_edge = params['Cost_mem_edge'] # Cost of Memory unit at the edge per hours
    global_Cost_cpu_cloud = params['Cost_cpu_cloud']   # Cost of CPU unit at the cloud per hours
    global_Cost_mem_cloud = params['Cost_mem_cloud']   # Cost of Memory unit at the cloud per hours
    global_Cost_network = params['Cost_network']   # Cost of network per GB

    
    # optional paramenters
    global_Di = params['Di'] if 'Di' in params else np.zeros(2*global_M)
    global_Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*global_M)
    global_Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*global_M)
    global_look_ahead = params['look_ahead'] if 'look_ahead' in params else 1 # look ahead factor to increase the delay decrease target
    global_cache_ttl = params['cache_ttl'] if 'cache_size' in params else 10 # cache expiry in round
    global_locked_b = params['locked_b'] if 'locked_b' in params else np.zeros(global_M) # binary encoding of microservice that can not be moved at the edge
    global_dp_builder = locals()[params['dp_builder']] if 'dp_builder' in params else dp_builder_with_traces # dependency path builder function
    global_S_cloud_old = np.ones(int(global_M)) # EPAMP assumes all microservice instances run in the cloud
    global_S_cloud_old[global_M-1] = 0 # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1 
    global_S_b_old = np.concatenate((global_S_cloud_old, global_S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    global_u_limit = params['u_limit'] if 'u_limit' in params else global_M # maximum number of microservices upgrade to consider in the single path adding greedy iteraction (lower reduce optimality but increase computaiton speed)
    global_traces = params['traces'] if 'traces' in params else None # flag to enable traces generation
    global_max_dps = params['max_dps'] if 'max_dps' in params else 1e6 # maximum number of dependency paths to consider in an optimization iteration
    global_max_traces = params['max_traces'] if 'max_traces' in params else 1024 # maximum number of traces to generate
    
    # Check if the graph is acyclic
    Fcm_unitary = np.where(global_Fcm > 0, 1, 0)
    global_G = nx.DiGraph(Fcm_unitary) # Create microservice dependency graph
    global_ms_distances = nx.floyd_warshall_numpy(global_G)
    if nx.is_directed_acyclic_graph(global_G)==False: 
        logger.critical(f"Microservice dependency graph is not acyclic, EPAMP optimization can not be used")
        result_edge=dict()
        result_edge['S_edge_b'] = global_S_b_old[global_M:].astype(int)
        result_edge['to-apply'] = list()
        result_edge['to-delete'] = list()
        result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(global_S_b_old[global_M:]==1))
        result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}, Cost: {result_edge['Cost']}, delay decrease: {result_edge['delay_decrease']}, cost increase: {result_edge['cost_increase']}"
        return result_edge 
    
    global_Rs = np.tile(global_Rs, 2)  # Expand the Rs vector to support matrix operations
    
    # SAVE INITIAL (OLD) METRICS VALUES ##
    global_Fci_old = np.matrix(buildFci(global_S_b_old, global_Fcm, global_M)) # (2*M,2*M) instance-set call frequency matrix
    global_Nci_old = computeNc(global_Fci_old, global_M, 2)  # (2*M,) number of instance call per user request
    global_delay_old,_,_,global_rhonce_old = computeDTot(global_S_b_old, global_Nci_old, global_Fci_old, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))  # Total delay of the current configuration. It includes only network delays
    global_Cost_old, global_Cost_old_edge,global_Cost_cpu_old_edge,global_Cost_mem_old_edge, global_Cost_old_cloud,global_Cost_cpu_old_cloud,global_Cost_mem_old_cloud, global_Cost_traffic_old = utils.computeCost(global_Acpu_old, global_Amem_old, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, global_rhonce_old * global_Ne, global_Cost_network)

    ## variables initialization ##
    S_b_temp = np.copy(global_S_b_old) # S_b_temp is the temporary placement state used in a greedy round
    S_b_new =  np.copy(global_S_b_old) # S_b_new is the new placement state 
    Acpu_new = np.copy(global_Acpu_old)    # Acpu_new is the new CPU request vector
    Amem_new = np.copy(global_Amem_old)   # Amem_new is the new Memory request vector
    rhoce_new = global_rhonce_old   # rhoce_new is the new cloud-edge network utilization
    Nci_new = np.copy(global_Nci_old)    # Nci_new is the new number of instance call per user request
    Acpu_temp = np.copy(global_Acpu_old)   # Acpu_temp is the temporary CPU request vector used in a greedy round
    Amem_temp = np.copy(global_Amem_old)   # Amem_temp is the temporary Memory request vector used in a greedy round
    S_b_opt = np.copy(global_S_b_old)  # S_b_opt is the best placement state computed by a greedy round
    Acpu_opt = np.copy(global_Acpu_old)  # Acpu_opt is the best CPU request vector computed by a greedy round
    Amem_opt = np.copy(global_Amem_old)  # Amem_opt is the best Memory request vector computed by a greedy round
    delay_opt = global_delay_old       # delay_opt is the best delay computed by a greedy round. It includes only network delays
    rhoce_opt = global_rhonce_old      # rhoce_opt is the best cloud-edge network utilization computed by a greedy round
    Fci_new = np.copy(global_Fci_old)    # Fci_new is the new instance-set call frequency matrix
    Fci_opt = np.copy(global_Fci_old)    # Fci_opt is the best instance-set call frequency matrix computed by a greedy round

    # result caches to accelerate computation
    
    global_cache = dict()    # cache dictionary for all caches
    global_cache['delay']=dict()  # cache for delay computation
    global_cache['rhoce']=dict()   # cache for rhoce computation
    global_cache['expire']=dict() # cache for expiration round
    global_cache['Acpu']=dict()   # cache for CPU request vector
    global_cache['Amem']=dict()   # cache for Memory request vector
    global_cache['Fci']=dict()    # cache for instance-set call frequency matrix
    global_cache['Nci']=dict()    # cache for number of instance call per user request
    global_cache['cost']=dict()   # cache for cost computation
    global_cache['cache_hit'] = 0  # cache hit counter
    global_cache['cache_access'] = 0   # cache access counter

    skip_delay_increase = False    # skip delay increase states to accelerate computation wheter possible
    if global_traces is None:
        dependency_paths_b_full_built = False # flag to check if the full dependency paths have been built
        dependency_paths_b_full = np.empty((0,global_M), int) # Storage of full set of binary-based (b) encoded dependency paths
    else:
        dependency_paths_b_full_built = True
        dependency_paths_b_full = global_traces


    ## Greedy addition of dependency paths ##
    logger.info(f"ADDING PHASE")
    round = -1
    dependency_paths_b_added = np.empty((0,global_M),dtype=int) # list of added dependency paths
    while True:
        round += 1
        logger.info(f'-----------------------')
        w_opt = float("inf") # Initialize the weight
        skip_delay_increase = False     # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)      # S_b_new is the new placement state
        np.copyto(Acpu_new,Acpu_opt)    # Acpu_new is the new CPU request vector, Acpu_opt is the best CPU request vector computed by the previos greedy round
        np.copyto(Amem_new,Amem_opt)    # Amem_new is the new Memory request vector, Amem_opt is the best Memory request vector computed by the previos greedy round
        np.copyto(Fci_new,Fci_opt)      # Fci_new is the new instance-set call frequency matrix, Fci_opt is the best instance-set call frequency matrix computed by the previos greedy round
        delay_new = delay_opt           # delay_new is the new delay. It includes only network delays
        rhoce_new = rhoce_opt           # rhoce_new is the new cloud-edge network utilization
        cost_new  = utils.computeCost(Acpu_new, Amem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new * global_Ne, global_Cost_network)[0] # Total  cost of the new configuration
        logger.info(f'new state {np.argwhere(S_b_new[global_M:]==1).squeeze()}, cost {cost_new}, delay decrease {1000*(global_delay_old-delay_new)} ms, cost increase {cost_new-global_Cost_old}')
        
        # Check if the delay reduction and other constraints are reached
        
        if global_delay_old-delay_new >= global_delay_decrease_target * global_look_ahead:
            # delay reduction reached
            logger.info(f'delay reduction reached')
            break

        # BUILDING OF COMPOSITE DEPENDENCY PATHS WITH SWEEPING
        Nci_new = computeNc(Fci_new, global_M, 2)
        dependency_paths_b = global_dp_builder(S_b_new, Acpu_new, Amem_new, Nci_new, round)
        
        if len(dependency_paths_b) == 0:
            # All dependency path considered no other way to reduce delay
            logger.info(f'All dependency path considered no other way to reduce delay')
            break

        ## GREEDY ROUND ##
        for dpi,path_b in enumerate(dependency_paths_b) :
            # merging path_b and S_b_new into S_b_temp
            path_n = np.argwhere(path_b.flatten()==1).squeeze() # numerical id of the microservices of the dependency path
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[global_M+path_n] = 1
            
            # cache probing
            _, result = evaluate_perf(S_b_temp, Acpu_new, Amem_new, Nci_new, round)
            delay_temp = result['delay']
            Acpu_temp = result['Acpu']
            Amem_temp = result['Amem']
            Fci_temp = result['Fci']
            rhoce_temp = result['rhoce']
            Cost_temp = result['cost']
            
            cost_increase_temp = Cost_temp - cost_new # cost increase wrt the new state 
            delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
            if skip_delay_increase and delay_decrease_temp<0:
                logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()} skipped for negative delay decrease')
                continue

            # weighting
            # r_delay_decrease = global_delay_decrease_target * global_look_ahead - (global_delay_old-delay_new) # residul delay to decrease wrt previous conf
            r_delay_decrease = 1e6  # test
            if delay_decrease_temp <= 0:  
                w = 1e6 + cost_increase_temp *  1000 * abs(delay_decrease_temp)
            else:
                w = cost_increase_temp /  max(min(1000*delay_decrease_temp, 1000*r_delay_decrease),1e-3) # 1e-3 used to avoid division by zero
                skip_delay_increase = True
            
            
            logger.debug(f'considered dependency path {np.argwhere(path_b[0]==1).flatten()}, cost increase {cost_increase_temp}, delay decrease {1000*delay_decrease_temp} ms, delay {delay_temp} ms, weight {w}')

            if w < w_opt:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                Acpu_opt = np.copy(Acpu_temp)
                Amem_opt = np.copy(Amem_temp)
                Fci_opt = np.copy(Fci_temp)
                delay_opt = delay_temp
                rhoce_opt = rhoce_temp
                w_opt = w
                dp_opt = path_b.copy().reshape(1,global_M)
                cost_opt = Cost_temp
       
        dependency_paths_b_added = np.append(dependency_paths_b_added,dp_opt,axis=0)
        logger.info(f"chache hit probability {global_cache['cache_hit']/(global_cache['cache_access'])}")
        
        if w_opt == inf:
            # no improvement possible in the greedy round
            logger.info(f'no improvement possible in the greedy round')
            break
        
        logger.info(f'added dependency path {np.argwhere(dp_opt==1)[:,1].flatten()}')  
 
        # cache cleaning
        cache_cleaning(round)

    

    # Remove leaves to reduce cost
    while True:
        w_opt = -1 # weight of the best removal
        leaf_best = -1 # index of the leaf microservice to remove
        S_b_temp = np.zeros(2*global_M)
        # try to remove leaves microservices
        Fci_new = np.matrix(buildFci(S_b_new, global_Fcm, global_M))
        Nci_new = computeNc(Fci_new, global_M, 2)
        S_b_new_a = np.array(S_b_new[global_M:]).reshape(global_M,1)
        delay_new,_,_,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))
        utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,global_Acpu_old,global_Amem_old,global_Nci_old)
        cost_new = utils.computeCost(Acpu_new, Amem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge,global_Cost_cpu_cloud, global_Cost_mem_cloud,rhoce_new*global_Ne,global_Cost_network)[0]
        edge_leaves = np.logical_and(np.sum(Fci_new[global_M:2*global_M-1,global_M:2*global_M-1], axis=1)==0, S_b_new_a[:global_M-1]==1) # edge microservice with no outgoing calls to other edge microservices
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        edge_leaves = edge_leaves+global_M # index of the edge microservice in the full state
        logger.info(f'pruning candidates {edge_leaves-global_M}')
        for leaf in edge_leaves:
            # try remove microservice
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[leaf] = 0
            Acpu_temp = np.zeros(2*global_M)
            Amem_temp = np.zeros(2*global_M)
            Fci_temp = np.matrix(buildFci(S_b_temp, global_Fcm, global_M))
            Nci_temp = computeNc(Fci_temp, global_M, 2)
            delay_temp,_,_,rhoce_temp = computeDTot(S_b_temp, Nci_temp, Fci_temp, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))
            delay_increase_temp = max(delay_temp - delay_new,1e-3)
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            Cost_temp = utils.computeCost(Acpu_temp, Amem_temp, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud,rhoce_temp*global_Ne, global_Cost_network)[0]
            cost_decrease = cost_new - Cost_temp
            w = cost_decrease/delay_increase_temp
            utils.computeResourceShift(Acpu_temp,Amem_temp,Nci_temp,Acpu_new,Amem_new,Nci_new)
            
            if w>w_opt and global_delay_old - delay_temp > global_delay_decrease_target:
                # possible removal
                w_opt = w
                leaf_best = leaf
                delay_reduction = global_delay_old - delay_temp
        if leaf_best>-1:
            logger.info(f'pruned microservice {leaf_best-global_M}, delay reduction: {delay_reduction}')
            S_b_new[leaf_best] = 0
        else:
            break
    
    # compute final values
    Fci_new = np.matrix(buildFci(S_b_new, global_Fcm, global_M))
    Nci_new = computeNc(Fci_new, global_M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, global_Di, global_Rs, global_RTT, global_Ne, global_lambd, global_M, np.empty(0))
    delay_decrease_new = global_delay_old - delay_new
    utils.computeResourceShift(Acpu_new,Amem_new,Nci_new,global_Acpu_old,global_Amem_old,global_Nci_old)
    cost_new, Cost_new_edge,Cost_cpu_new_edge,Cost_mem_new_edge, Cost_new_cloud,Cost_cpu_new_cloud,Cost_mem_new_cloud, Cost_traffic_new = utils.computeCost(Acpu_new, Amem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new * global_Ne, global_Cost_network) # Total cost of new state
    cost_increase_new = cost_new - global_Cost_old

    result_edge = dict()
    
    # extra information
    result_edge['S_edge_b'] = S_b_new[global_M:].astype(int)
    result_edge['Cost'] = cost_new
    result_edge['Cost_edge'] = Cost_new_edge
    result_edge['Cost_cpu_edge'] = Cost_cpu_new_edge
    result_edge['Cost_mem_edge'] = Cost_mem_new_edge
    result_edge['Cost_cloud'] = Cost_new_cloud
    result_edge['Cost_cpu_cloud'] = Cost_cpu_new_cloud
    result_edge['Cost_mem_cloud'] = Cost_mem_new_cloud
    result_edge['Cost_traffic'] = Cost_traffic_new
    result_edge['delay_decrease'] = delay_decrease_new
    result_edge['cost_increase'] = cost_increase_new
    result_edge['Acpu'] = Acpu_new
    result_edge['Amem'] = Amem_new
    result_edge['Fci'] = Fci_new
    result_edge['Nci'] = Nci_new
    result_edge['delay'] = delay_new
    result_edge['di'] = di_new
    result_edge['dn'] = dn_new
    result_edge['rhoce'] = rhoce_new
    
    # required return information
     
    result_cloud = dict()
    result_cloud['to-apply'] = list()
    result_cloud['to-delete'] = list()
    result_cloud['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[:global_M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"


    result_edge['to-apply'] = utils.numpy_array_to_list(np.argwhere(S_b_new[global_M:]-global_S_b_old[global_M:]>0))
    result_edge['to-delete'] = utils.numpy_array_to_list(np.argwhere(global_S_b_old[global_M:]-S_b_new[global_M:]>0))
    result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[global_M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_edge['delay_decrease'] < global_delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    return result_return

