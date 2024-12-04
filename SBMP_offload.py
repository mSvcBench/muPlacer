# pylint: disable=C0103, C0301
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import networkx as nx
from computeN import computeN
from buildFi import buildFi
from numpy import inf
from computeDTot import computeDTot
import logging
import sys
import utils
from S2id import S2id
from id2S import id2S
import time


np.seterr(divide='ignore', invalid='ignore')

# Set up logger
logger = logging.getLogger('SBMP_offload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s SBMP offload %(levelname)s %(message)s'))
logger.propagate = False

def sbmp_o(params):

   

    def cache_probe(S_b, round):
        result=dict()
        result['delay'] = None
        result['Ucpu'] = None
        result['Umem'] = None
        result['Fi'] = None
        result['N'] = None
        result['rhoce'] = None
        result['cost'] = None
        hit = False
        global_cache['cache_access'] += 1
        S_id_edge=np.array2string(S_b[global_M:])
        if S_id_edge in global_cache['delay']:
            logger.debug(f'cache_hit for {np.argwhere(S_b[global_M:]==1).squeeze()}')
            global_cache['cache_hit'] += 1
            result['delay'] = global_cache['delay'][S_id_edge]
            result['Ucpu'] = np.copy(global_cache['Ucpu'][S_id_edge])
            result['Umem'] = np.copy(global_cache['Umem'][S_id_edge])
            result['Fi'] = global_cache['Fi'][S_id_edge].copy()
            result['N'] = global_cache['N'][S_id_edge].copy()
            result['rhoce'] = global_cache['rhoce'][S_id_edge]
            result['cost'] = global_cache['cost'][S_id_edge]
            global_cache['expire'][S_id_edge] = round
            hit = True
        return hit, result
    
    def evaluate_perf(S_b_new, Ucpu_old, Umem_old, N_old, round):
        hit, result = cache_probe(S_b_new, round)
        if not hit:
            Ucpu_new = np.zeros(2*global_M)
            Umem_new = np.zeros(2*global_M)
            Fi_new = np.matrix(buildFi(S_b_new, global_Fm, global_M))
            N_new = computeN(Fi_new, global_M, 2)
            delay_new,_,_,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, global_Di, global_L, global_RTT, global_B, global_lambd, global_M, np.empty(0))
            utils.computeResourceShift(Ucpu_new, Umem_new,N_new,Ucpu_old,Umem_old,N_old)
            cost_new = utils.computeCost(Ucpu_new, Umem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new*global_B, global_Cost_network)[0] # Total  cost of the temp state
            # cache insertion
            cache_insert(S_b_new, delay_new, Ucpu_new, Umem_new, Fi_new, N_new, rhoce_new, cost_new, round)
            result = dict()
            result['delay'] = delay_new
            result['Ucpu'] = Ucpu_new.copy()
            result['Umem'] = Umem_new.copy()
            result['Fi'] = Fi_new.copy()
            result['N'] = N_new.copy()
            result['rhoce'] = rhoce_new
            result['cost'] = cost_new
        return hit, result
    
    def cache_insert(S_b, delay, Ucpu, Umem, Fi, N, rhoce, cost, round):
        S_id_edge=np.array2string(S_b[global_M:])
        global_cache['delay'][S_id_edge] = delay
        global_cache['rhoce'][S_id_edge] = rhoce
        global_cache['Ucpu'][S_id_edge]=np.copy(Ucpu)
        global_cache['Umem'][S_id_edge]=np.copy(Umem)
        global_cache['expire'][S_id_edge] = round
        global_cache['Fi'][S_id_edge]=Fi.copy()
        global_cache['N'][S_id_edge]=N.copy()
        global_cache['cost'][S_id_edge]=cost
        logger.debug(f'cache insert for {np.argwhere(S_b_temp[global_M:]==1).squeeze()}')
    
    def cache_cleaning(round):
        for key in list(global_cache['delay'].keys()):
            if global_cache['expire'][key] + global_cache_ttl < round:
                del global_cache['delay'][key]
                del global_cache['rhoce'][key]
                del global_cache['Ucpu'][key]
                del global_cache['Umem'][key]
                del global_cache['expire'][key]
    
    def sgs_builder_with_single_path_adding(S_b_init, Ucpu_init, Umem_init, N_init, round):
        ## BUILDING OF EXPANDING SUBGRAPH WITH SINGLE PATH ADDING - PAMP Algorithm ##
        
        nonlocal expanding_subgraphs_b_full_built, expanding_subgraphs_b_full
        if not expanding_subgraphs_b_full_built:
            for ms in range(global_M-1):
                paths_n = list(nx.all_simple_paths(global_G, source=global_M-1, target=ms)) 
                for path_n in paths_n:
                    # path_n numerical id (n) of the microservices of the expanding subgraph
                    # If all microservices in the subgraph are in the edge this subgraph can not extend the edge graph
                    if all(global_S_b_old[global_M+np.array([path_n])].squeeze()==1):
                        continue
                    else:
                        path_b = np.zeros((1,global_M),int)
                        path_b[0,path_n] = 1 # Binary-based (b) encoding of the expanding subgraph
                        expanding_subgraphs_b_full = np.append(expanding_subgraphs_b_full,path_b,axis=0)
            expanding_subgraphs_b_full_built = True
        residual = np.sum(np.maximum(expanding_subgraphs_b_full-S_b_init[global_M:],0),axis=1)
        rl = np.argwhere((residual > 0) & (residual <= global_expanding_depth)).flatten()
        return expanding_subgraphs_b_full[rl]
    
    def sgs_builder_traces(S_b_init, Ucpu_init, Umem_init, N_init, round):
        ## BUILDING OF EXPANDING SUBGRAPHS FROM TRACES#
        nonlocal expanding_subgraphs_b_full_built, expanding_subgraphs_b_full
        _,result = evaluate_perf(S_b_init, Ucpu_init, Umem_init, N_init, round)
        Fi_init = result['Fi']
        if not expanding_subgraphs_b_full_built:
            expanding_subgraphs_b_full = utils.sgs_builder_traces_full(global_M,global_max_traces,global_Fm)
            # n_traces = global_max_traces
            # expanding_subgraphs_b_full = np.empty((0,global_M), int)
            # user = global_M-1
            # iteration = 0
            # while True:
            #     iteration += 1
            #     trace_sample_b = np.zeros(global_M)
            #     trace_sample_b = sgs_builder_trace(user,trace_sample_b,global_Fm)
            #     expanding_subgraphs_b_full = np.append(expanding_subgraphs_b_full, trace_sample_b.reshape(1, -1), axis=0)
            #     if len(expanding_subgraphs_b_full) >= n_traces or (iteration > 100*n_traces and len(expanding_subgraphs_b_full) > 20):
            #         break
            # trace_sample_b = np.ones(global_M)  # add full edge trace
            # expanding_subgraphs_b_full = np.append(expanding_subgraphs_b_full, trace_sample_b.reshape(1, -1), axis=0)
            expanding_subgraphs_b_full_built = True
        
        expanding_subgraphs_b = np.empty((0,global_M), int)
        # remove traces fully in the edge
        residual = np.argwhere(np.sum(np.maximum(expanding_subgraphs_b_full-S_b_init[global_M:],0),axis=1)>0).flatten()
        expanding_subgraphs_b = expanding_subgraphs_b_full[residual]
        
        # clean from these traces the cloud microservices that are at a distance greather than expanding_depth from the edge gateways
        edge_leaves = np.unique(np.argwhere(Fi_init[global_M:2*global_M,0:global_M]>0)[:,0]) # list of edge graph leaves: microservices in the edge with at least one call from the cloud
        allowed_cloud_ms = np.empty((0), int)
        for edge_gw in edge_leaves:
            allowed_cloud_ms = np.append(allowed_cloud_ms, np.argwhere(global_ms_distances[edge_gw][:] <= global_expanding_depth).flatten())
        allowed_cloud_ms = np.unique(allowed_cloud_ms)
        not_allowed_ms = np.setdiff1d(np.arange(global_M), allowed_cloud_ms)
        
        expanding_subgraphs_b[:,not_allowed_ms]=0
        
        # remove microservice looked by the user
        expanding_subgraphs_b[:,np.argwhere(global_locked_b>0)]=0

        # remove traces fully in the edge
        residual = np.argwhere(np.sum(np.maximum(expanding_subgraphs_b-S_b_init[global_M:],0),axis=1)>0).flatten()
        expanding_subgraphs_b = expanding_subgraphs_b[residual]

        # compute the frequency of the expanding subgraph paths to return the most frequently used
        expanding_subgraphs_b, paths_freq = np.unique(expanding_subgraphs_b, axis=0,return_counts=True)
        mfu_expanding_subgraph_id = np.flip(np.argsort(paths_freq))

        return expanding_subgraphs_b[mfu_expanding_subgraph_id[:min(global_max_sgs,len(mfu_expanding_subgraph_id))]]

    # def sgs_builder_trace(node,trace,global_Fm):
    #     children = np.argwhere(global_Fm[node,0:global_M]>0).flatten()
    #     for child in children:
    #         if np.random.random() < global_Fm[node,child]:
    #             trace[child] = 1
    #             trace = sgs_builder_trace(child,trace,global_Fm)
    #     return trace

    ## INITIALIZE VARIABLES ##

    # mandatory paramenters
    global_S_edge_old = params['S_edge_b'] # (M,) binary presence at the edge
    global_Ucpu_old = params['Ucpu'] # (2*M,) CPU usage vector
    global_Umem_old = params['Umem'] # (2*M,) Memory usage vector
    global_Fm = params['Fm'] # (M,M) microservice call frequency matrix
    global_M = params['M'] # number of microservices
    global_lambd = params['lambd'] # user request rate vector
    global_L = params['L'] # (M,) response length vector
    global_delay_decrease_target = params['delay_decrease_target'] # target delay decrease
    global_RTT = params['RTT'] # RTT between edge and cloud
    global_B = params['B'] # network bandwidth
    global_Cost_cpu_edge = params['Cost_cpu_edge'] # Cost of CPU unit at the edge per hours
    global_Cost_mem_edge = params['Cost_mem_edge'] # Cost of Memory unit at the edge per hours
    global_Cost_cpu_cloud = params['Cost_cpu_cloud']   # Cost of CPU unit at the cloud per hours
    global_Cost_mem_cloud = params['Cost_mem_cloud']   # Cost of Memory unit at the cloud per hours
    global_Cost_network = params['Cost_network']   # Cost of network per GB

    # optional paramenters
    global_Di = params['Di'] if 'Di' in params else np.zeros(2*global_M) # (2*M,) interna delay vector
    global_Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*global_M) # (2*M,) memory quota vector (Kubernetes CPU Request)
    global_Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*global_M) # (2*M,) CPU quota vector (Kubernetes MEM Request)
    global_cache_ttl = params['cache-ttl'] if 'cache-ttl' in params else 10 # cache expiry in round
    global_locked_b = params['locked_b'] if 'locked_b' in params else np.zeros(global_M) # binary encoding of microservice that can not be moved at the edge
    global_sgs_builder = locals()[params['sgs-builder']] if 'sgs-builder' in params else locals()['sgs_builder_traces'] # expanding subgraph builder function
    global_expanding_depth = params['expanding-depth'] if 'expanding-depth' in params else global_M # maximum number of microservices upgrade to consider in the single path adding greedy iteraction (lower reduce optimality but increase computaiton speed)
    global_traces_b = params['traces-b'] if 'traces-b' in params else None # flag to enable traces generation
    global_max_sgs = params['max-sgs'] if 'max-sgs' in params else 1e6 # maximum number of subgraphs to consider in an optimization iteration
    global_max_traces = params['max-traces'] if 'max-traces' in params else 1024 # maximum number of traces to generate
    global_delay_decrease_stop_condition = params['delay_decrease_stop_condition'] if 'delay_decrease_stop_condition' in params else global_delay_decrease_target # delay decrease early stop
    global_HPA_cpu_th = params['HPA_cpu_th'] if 'HPA_cpu_th' in params else None # CPU threshold for HPA
    
    
    global_S_cloud_old = np.ones(int(global_M)) # SBMP assumes all microservice instances run in the cloud
    global_S_cloud_old[global_M-1] = 0 # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1 
    global_S_b_old = np.concatenate((global_S_cloud_old, global_S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge
    # Check if the graph is acyclic
    Fm_unitary = np.where(global_Fm > 0, 1, 0)
    global_G = nx.DiGraph(Fm_unitary) # Create microservice dependency graph
    global_ms_distances = nx.floyd_warshall_numpy(global_G)
    if nx.is_directed_acyclic_graph(global_G)==False: 
        logger.critical(f"Microservice dependency graph is not acyclic, SBMP optimization can not be used")
        result_edge=dict()
        result_edge['S_edge_b'] = global_S_b_old[global_M:].astype(int)
        result_edge['to-apply'] = list()
        result_edge['to-delete'] = list()
        result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(global_S_b_old[global_M:]==1))
        result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}, Cost: {result_edge['Cost']}, delay decrease: {result_edge['delay_decrease']}, cost increase: {result_edge['cost_increase']}"
        return result_edge 
    
    global_L = np.tile(global_L, 2)  # Expand the Rs vector to support matrix operations
    
    # SAVE INITIAL (OLD) METRICS VALUES ##
    global_Fi_old = np.matrix(buildFi(global_S_b_old, global_Fm, global_M)) # (2*M,2*M) instance-set call frequency matrix
    global_N_old = computeN(global_Fi_old, global_M, 2)  # (2*M,) number of instance call per user request
    global_delay_old,_,_,global_rhonce_old = computeDTot(global_S_b_old, global_N_old, global_Fi_old, global_Di, global_L, global_RTT, global_B, global_lambd, global_M, np.empty(0))  # Total delay of the current configuration. It includes only network delays
    global_Cost_old, global_Cost_old_edge,global_Cost_old_cloud,global_Cost_traffic_old = utils.computeCost(global_Ucpu_old, global_Umem_old, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, global_rhonce_old * global_B, global_Cost_network, global_HPA_cpu_th)

    ## variables initialization ##
    S_b_temp = np.copy(global_S_b_old) # S_b_temp is the temporary placement state used in a greedy round
    S_b_new =  np.copy(global_S_b_old) # S_b_new is the new placement state 
    Ucpu_new = np.copy(global_Ucpu_old)    # Ucpu_new is the new CPU usage vector
    Umem_new = np.copy(global_Umem_old)   # Umem_new is the new Memory usage vector
    rhoce_new = global_rhonce_old   # rhoce_new is the new cloud-edge network utilization
    N_new = np.copy(global_N_old)    # N_new is the new number of instance call per user request
    Ucpu_temp = np.copy(global_Ucpu_old)   # Ucpu_temp is the temporary CPU usage vector used in a greedy round
    Umem_temp = np.copy(global_Umem_old)   # Umem_temp is the temporary Memory usage vector used in a greedy round
    S_b_opt = np.copy(global_S_b_old)  # S_b_opt is the best placement state computed by a greedy round
    Ucpu_opt = np.copy(global_Ucpu_old)  # Ucpu_opt is the best CPU usage vector computed by a greedy round
    Umem_opt = np.copy(global_Umem_old)  # Umem_opt is the best Memory usage vector computed by a greedy round
    delay_opt = global_delay_old       # delay_opt is the best delay computed by a greedy round. It includes only network delays
    rhoce_opt = global_rhonce_old      # rhoce_opt is the best cloud-edge network utilization computed by a greedy round
    Fi_new = np.copy(global_Fi_old)    # Fi_new is the new instance-set call frequency matrix
    Fi_opt = np.copy(global_Fi_old)    # Fi_opt is the best instance-set call frequency matrix computed by a greedy round

    # result caches to accelerate computation
    
    global_cache = dict()    # cache dictionary for all caches
    global_cache['delay']=dict()  # cache for delay computation
    global_cache['rhoce']=dict()   # cache for rhoce computation
    global_cache['expire']=dict() # cache for expiration round
    global_cache['Ucpu']=dict()   # cache for CPU usage vector
    global_cache['Umem']=dict()   # cache for Memory usage vector
    global_cache['Fi']=dict()    # cache for instance-set call frequency matrix
    global_cache['N']=dict()    # cache for number of instance call per user request
    global_cache['cost']=dict()   # cache for cost computation
    global_cache['cache_hit'] = 0  # cache hit counter
    global_cache['cache_access'] = 0   # cache access counter

    skip_delay_increase = False    # skip delay increase states to accelerate computation wheter possible
    if global_traces_b is None:
        expanding_subgraphs_b_full_built = False # flag to check if the full expanding subgraph set has been built
        expanding_subgraphs_b_full = np.empty((0,global_M), int) # Storage of full set of binary-based (b) encoded expanding subraphs
    else:
        expanding_subgraphs_b_full_built = True
        expanding_subgraphs_b_full = global_traces_b


    ## Greedy addition of expanding subgraphs paths ##
    logger.info(f"ADDING PHASE")
    round = -1
    expanding_subgraphs_b_added = np.empty((0,global_M),dtype=int) # list of added expanding subgraphs
    while True:
        round += 1
        logger.info(f'-----------------------')
        w_opt = float("inf") # Initialize the weight
        skip_delay_increase = False     # Skip negative weight to accelerate computation
        np.copyto(S_b_new,S_b_opt)      # S_b_new is the new placement state
        np.copyto(Ucpu_new,Ucpu_opt)    # Ucpu_new is the new CPU usage vector, Ucpu_opt is the best CPU usage vector computed by the previos greedy round
        np.copyto(Umem_new,Umem_opt)    # Umem_new is the new Memory usage vector, Umem_opt is the best Memory usage vector computed by the previos greedy round
        np.copyto(Fi_new,Fi_opt)      # Fi_new is the new instance-set call frequency matrix, Fi_opt is the best instance-set call frequency matrix computed by the previos greedy round
        delay_new = delay_opt           # delay_new is the new delay. It includes only network delays
        rhoce_new = rhoce_opt           # rhoce_new is the new cloud-edge network utilization
        Cost_new  = utils.computeCost(Ucpu_new, Umem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new * global_B, global_Cost_network,global_HPA_cpu_th)[0] # Total  cost of the new configuration
        logger.info(f'new state {np.argwhere(S_b_new[global_M:]==1).squeeze()}, cost {Cost_new}, delay decrease {1000*(global_delay_old-delay_new)} ms, cost increase {Cost_new-global_Cost_old}')
        
        # Check if the delay reduction and other constraints are reached
        
        if global_delay_old-delay_new >= global_delay_decrease_stop_condition:
            # delay reduction reached
            logger.info(f'delay reduction stop value reached')
            if global_delay_old-delay_new >= global_delay_decrease_target:
                # delay reduction reached
                logger.info(f'target delay reduction reached')
            break

        # BUILDING OF EXPANDING SUBGRAPH
        N_new = computeN(Fi_new, global_M, 2)
        expanding_subgraphs_b = global_sgs_builder(S_b_new, Ucpu_new, Umem_new, N_new, round)
        
        if len(expanding_subgraphs_b) == 0:
            # All expanding subgraph considered no other way to reduce delay
            logger.info(f'All expanding subgraphs considered no other way to reduce delay')
            break

        ## GREEDY ROUND ##
        for _,subgraph_b in enumerate(expanding_subgraphs_b) :
            # merging path_b and S_b_new into S_b_temp
            subgraph_n = np.argwhere(subgraph_b.flatten()==1).squeeze() # numerical id of the microservices of the expanding subgraph
            np.copyto(S_b_temp, S_b_new)
            S_b_temp[global_M+subgraph_n] = 1
            
            # cache probing
            _, result = evaluate_perf(S_b_temp, Ucpu_new, Umem_new, N_new, round)
            delay_temp = result['delay']
            Ucpu_temp = result['Ucpu']
            Umem_temp = result['Umem']
            Fi_temp = result['Fi']
            rhoce_temp = result['rhoce']
            Cost_temp = result['cost']
            
            cost_increase_temp = Cost_temp - Cost_new # cost increase wrt the new state 
            delay_decrease_temp = delay_new - delay_temp    # delay reduction wrt the new state
            if skip_delay_increase and delay_decrease_temp<0:
                logger.debug(f'considered expanding subgrah {np.argwhere(subgraph_b[0]==1).flatten()} skipped for negative delay decrease')
                continue

            # weighting
            r_delay_decrease = 1e6  # test
            if delay_decrease_temp <= 0:  
                w = 1e6 + cost_increase_temp *  1000 * abs(delay_decrease_temp)
            else:
                w = cost_increase_temp /  max(min(1000*delay_decrease_temp, 1000*r_delay_decrease),1e-3) # 1e-3 used to avoid division by zero
                skip_delay_increase = True
            
            
            logger.debug(f'considered expanding subgraph {np.argwhere(subgraph_b[0]==1).flatten()}, cost increase {cost_increase_temp}, delay decrease {1000*delay_decrease_temp} ms, delay {delay_temp} ms, weight {w}')

            if w < w_opt:
                # update best state of the greedy round
                np.copyto(S_b_opt,S_b_temp)
                Ucpu_opt = np.copy(Ucpu_temp)
                Umem_opt = np.copy(Umem_temp)
                Fi_opt = np.copy(Fi_temp)
                delay_opt = delay_temp
                rhoce_opt = rhoce_temp
                w_opt = w
                sg_opt = subgraph_b.copy().reshape(1,global_M)
                cost_opt = Cost_temp
       
        expanding_subgraphs_b_added = np.append(expanding_subgraphs_b_added,sg_opt,axis=0)
        logger.info(f"chache hit probability {global_cache['cache_hit']/(global_cache['cache_access'])}")
        
        if w_opt == inf:
            # no improvement possible in the greedy round
            logger.info(f'no improvement possible in the greedy round')
            break
        
        logger.info(f'added expanding subgraph {np.argwhere(sg_opt==1)[:,1].flatten()}')  
 
        # cache cleaning
        cache_cleaning(round)

    

    # Remove leaves to reduce cost
    while True:
        w_opt = -1 # weight of the best removal
        leaf_best = -1 # index of the leaf microservice to remove
        S_b_temp = np.zeros(2*global_M)
        # try to remove leaves microservices
        Fi_new = np.matrix(buildFi(S_b_new, global_Fm, global_M))
        N_new = computeN(Fi_new, global_M, 2)
        S_b_new_a = np.array(S_b_new[global_M:]).reshape(global_M,1)
        delay_new,_,_,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, global_Di, global_L, global_RTT, global_B, global_lambd, global_M, np.empty(0))
        utils.computeResourceShift(Ucpu_new,Umem_new,N_new,global_Ucpu_old,global_Umem_old,global_N_old)
        Cost_new = utils.computeCost(Ucpu_new, Umem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge,global_Cost_cpu_cloud, global_Cost_mem_cloud,rhoce_new*global_B,global_Cost_network,global_HPA_cpu_th)[0]
        edge_leaves = np.logical_and(np.sum(Fi_new[global_M:2*global_M-1,global_M:2*global_M-1], axis=1)==0, S_b_new_a[:global_M-1]==1) # edge microservice with no outgoing calls to other edge microservices
        edge_leaves = np.argwhere(edge_leaves)[:,0]
        edge_leaves = edge_leaves+global_M # index of the edge microservice in the full state
        logger.info(f'pruning candidates {edge_leaves-global_M}')
        for leaf in edge_leaves:
            # try remove microservice
            np.copyto(S_b_temp,S_b_new)
            S_b_temp[leaf] = 0
            Ucpu_temp = np.zeros(2*global_M)
            Umem_temp = np.zeros(2*global_M)
            Fi_temp = np.matrix(buildFi(S_b_temp, global_Fm, global_M))
            N_temp = computeN(Fi_temp, global_M, 2)
            delay_temp,_,_,rhoce_temp = computeDTot(S_b_temp, N_temp, Fi_temp, global_Di, global_L, global_RTT, global_B, global_lambd, global_M, np.empty(0))
            delay_increase_temp = max(delay_temp - delay_new,1e-3)
            utils.computeResourceShift(Ucpu_temp,Umem_temp,N_temp,Ucpu_new,Umem_new,N_new)
            Cost_temp = utils.computeCost(Ucpu_temp, Umem_temp, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud,rhoce_temp*global_B, global_Cost_network, global_HPA_cpu_th)[0]
            cost_decrease = Cost_new - Cost_temp
            w = cost_decrease/delay_increase_temp
            utils.computeResourceShift(Ucpu_temp,Umem_temp,N_temp,Ucpu_new,Umem_new,N_new)
            
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
    Fi_new = np.matrix(buildFi(S_b_new, global_Fm, global_M))
    N_new = computeN(Fi_new, global_M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, N_new, Fi_new, global_Di, global_L, global_RTT, global_B, global_lambd, global_M, np.empty(0))
    delay_decrease_new = global_delay_old - delay_new
    utils.computeResourceShift(Ucpu_new,Umem_new,N_new,global_Ucpu_old,global_Umem_old,global_N_old)
    Cost_new, Cost_new_edge, Cost_new_cloud, Cost_traffic_new = utils.computeCost(Ucpu_new, Umem_new, global_Qcpu, global_Qmem, global_Cost_cpu_edge, global_Cost_mem_edge, global_Cost_cpu_cloud, global_Cost_mem_cloud, rhoce_new * global_B, global_Cost_network, global_HPA_cpu_th) # Total cost of new state
    cost_increase_new = Cost_new - global_Cost_old

    result_metrics = dict()
    
    # extra information
    result_metrics['S_edge_b'] = S_b_new[global_M:].astype(int)
    result_metrics['Cost'] = Cost_new
    result_metrics['Cost_edge'] = Cost_new_edge
    result_metrics['Cost_cloud'] = Cost_new_cloud
    result_metrics['Cost_traffic'] = Cost_traffic_new
    result_metrics['delay_decrease'] = delay_decrease_new
    result_metrics['cost_increase'] = cost_increase_new
    result_metrics['Ucpu'] = Ucpu_new
    result_metrics['Umem'] = Umem_new
    result_metrics['Fi'] = Fi_new
    result_metrics['N'] = N_new
    result_metrics['delay'] = delay_new
    result_metrics['di'] = di_new
    result_metrics['dn'] = dn_new
    result_metrics['rhoce'] = rhoce_new
    
    # required return information
     
    result_cloud = dict()
    result_cloud['to-apply'] = list()
    result_cloud['to-delete'] = list()
    result_cloud['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[:global_M]==1))
    result_cloud['info'] = f"Result for offload - cloud microservice ids: {result_cloud['placement']}"

    result_edge = dict()
    result_edge['to-apply'] = utils.numpy_array_to_list(np.argwhere(S_b_new[global_M:]-global_S_b_old[global_M:]>0))
    result_edge['to-delete'] = utils.numpy_array_to_list(np.argwhere(global_S_b_old[global_M:]-S_b_new[global_M:]>0))
    result_edge['placement'] = utils.numpy_array_to_list(np.argwhere(S_b_new[global_M:]==1))

    result_edge['info'] = f"Result for offload - edge microservice ids: {result_edge['placement']}"

    if result_metrics['delay_decrease'] < global_delay_decrease_target:
        logger.warning(f"offload: delay decrease target not reached")
    
    result_return=list()
    result_return.append(result_cloud)  
    result_return.append(result_edge)
    result_return.append(result_metrics)
    return result_return

