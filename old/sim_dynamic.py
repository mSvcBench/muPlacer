import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(f'{parent_dir}/utils')
sys.path.append(f'{parent_dir}/strategies')

from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

from SBMP_offload import sbmp_o
from SBMP_unoffload import sbmp_u
from MFU import mfu
from IA import IA_heuristic
from igraph import *
from utils import buildFi, computeN, computeDTot, computeCost
from scipy.io import savemat
from numpy import inf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import logging

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

logging.basicConfig(stream=sys.stdout, level='ERROR',format='%(asctime)s GMA %(levelname)s %(message)s')

seed = 150273
np.random.seed(seed)
random.seed(seed)

res=np.array([])
trials = 10
RTT = 0.06    # RTT edge-cloud
M = 101 # n. microservices
Ne = 1e9    # bitrate cloud-edge

offload_threshold = 0.200 # threshold to offload in sec
unoffload_threshold = 0.190 # threshold to unoffload in sec
target_delay = unoffload_threshold + (offload_threshold-unoffload_threshold)/2.0 # target user delay (sec)

S_edge_b = np.zeros(M)  # initial state. 
S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
S_b[M-1] = 0  # User is not in the cloud

Fcm_range_min = 0.1 # min value of microservice call frequency 
Fcm_range_max = 0.5 # max value of microservice call frequency 
Acpu_range_min = 1  # min value of actual CPU consumption per instance-set
Acpu_range_max = 128 # max value of actual CPU consumption per instance-set
Rs_range_min = 200000 # min value of response size in bytes
Rs_range_max = 2000000   # max of response size in bytes

Cost_cpu_edge = 1.3 # cost of CPU at the edge
Cost_mem_edge = 1.3 # cost of memory at the edge
Cost_cpu_cloud = 1.0 # cost of CPU at the edge
Cost_mem_cloud = 1.0 # cost of memory at the edge

lambda_min = 10   # min user request rate (req/s)
lambda_max = 500   # max user request rate (req/s)
lambda_step = 10   # user request rate step (req/s)
lambda_range = list(range(lambda_min, lambda_max+lambda_step, lambda_step))  # user request rates (req/s)
lambda_range = lambda_range + list(range(lambda_max, lambda_min-lambda_step, -lambda_step))  # user request rates (req/s)

graph_algorithm = 'barabasi'

barabasi=dict()
barabasi['n'] = M-1
# barabasi['m'] = 1
# barabasi['power'] = 0.05
# barabasi['zero_appeal'] = 0.01
barabasi['m'] = 1
barabasi['power'] = 0.9
barabasi['zero_appeal'] = 3.65
random=dict()
random['n_parents'] = 3

random=dict()
random['n_parents'] = 3

Di_range_min = 0.005 # min value of internal delay (sec)
Di_range_max = 0.01 # max value of internal delay (sec)

max_algotithms = 10

show_graph = False
show_plot = False

cost_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of costs obtained by different algorithms 
cost_v_edge = np.empty((trials,len(lambda_range),max_algotithms)) # vector of costs obtained by different algorithms 
cost_v_cloud = np.empty((trials,len(lambda_range),max_algotithms)) # vector of costs obtained by different algorithms 
delay_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of delta delay obtained by different algorithms  
delta_cost_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of delta cost obtained by different algorithms  
p_time_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of processing time obtained by different algorithms
rhoce_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of rhoce obtained by different algorithms
edge_ms_v = np.empty((trials,len(lambda_range),max_algotithms)) # vector of number of edge microservice obtained by different algorithms

for t in range(trials):
    print(f"Trial {t}")
    
    # build dependency graph
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    if graph_algorithm == 'barabasi':
        g = Graph.Barabasi(n=barabasi['n'],m=barabasi['m'],power=barabasi['power'],zero_appeal=barabasi['zero_appeal'], directed=True)
        edges_reversal(g)
        Fcm[:M-1,:M-1] = np.matrix(g.get_adjacency()) 
    elif graph_algorithm == 'random':
        n_parents = 3
        for i in range(1,M-1):
            n_parent=np.random.randint(1,random['n_parents'])
            for j in range(n_parent):
                a = np.random.randint(i)
                Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(Fcm_range_min,Fcm_range_max) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)
    
    # set random values for actual CPU consumption per instance-set
    Acpu_void = np.random.uniform(Acpu_range_min,Acpu_range_max,M)   # actual CPU consumption per instance-set
    Acpu_void[M-1]=0 # user has no CPU consumption
    Amem_void = np.zeros(M) # do not account memory consumption
    Acpu_void = np.append(Acpu_void, np.zeros(M))
    Amem_void = np.append(Amem_void, np.zeros(M))
    # set random values for CPU and memory requests
    Qcpu = np.ones(2*M) # CPU quota
    Qmem = np.ones(2*M) # memory quota
    # set random values for response size
    Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # user has no response size
    # set random  internal delay
    Di = np.random.uniform(Di_range_min,Di_range_max,M)
    Di[M-1] = 0 # user has no internal delay
    Di = np.tile(Di, 2)
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fci_void = np.matrix(buildFi(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeN(Fci_void, M, 2)    # number of instance call per user request of the void state
    Cost_void = computeCost(Acpu_void, Amem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud)[0]# Total cost of the void state  
    if show_graph:
        G = nx.DiGraph(Fcm)
        nx.draw(G,with_labels=True)
        plt.show()

    S_b_old_dict = dict()
    for a in range(max_algotithms):
        S_b_old_dict[a] = S_b_void.copy()

    alg_type = [""] * max_algotithms # vector of strings describing algorithms used for te tests

    # full edge feasibility check
    S_b_full_e = np.ones(2*M)
    S_b_full_e[M-1]=0
    Acpu_full_e = np.zeros(2*M)
    Amem_full_e = np.zeros(2*M)
    Fci_full_e = np.matrix(buildFi(S_b_full_e, Fcm, M))
    Nci_full_e = computeN(Fci_full_e, M, 2)
    delay_full_e = computeDTot(S_b_full_e, Nci_full_e, Fci_full_e, Di, np.tile(Rs,2), RTT, Ne, lambda_max, M, np.empty(0))[0]
    if delay_full_e > offload_threshold:
        print(f"Full edge delay {delay_full_e} sec greather than offload threshold {offload_threshold} sec, simulation aborted")
        exit(1) 

    for lambda_i,lambda_val in enumerate(lambda_range):
        a=-1
        print(f'\n lambda {lambda_val} req/s')

        ## E_PAMP limit 2 ##
        a+=1
        alg_type[a] = "E_PAMP with sweeping"
        S_b_new = S_b_old_dict[a].copy()
        Acpu_new = np.zeros(2*M)
        Amem_new = np.zeros(2*M)
        Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
        Nci_new = computeN(Fci_new, M, 2)
        computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_void,Amem_void,Nci_void)
        delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, np.tile(Rs,2), RTT, Ne, lambda_val, M, np.empty(0))[0] # Total delay of the temp state. It includes only network delays
        if delay_new > offload_threshold:
            delay_decrease_target = delay_new - target_delay
            delay_increase_target = 0
            mode = 'offload'
            opt=True
        elif delay_new < unoffload_threshold and np.sum(S_b_new[M:2*M-1]):
            delay_decrease_target = 0
            delay_increase_target = target_delay - delay_new
            mode = 'unoffload'
            opt=True
        else:
            print(f"Delay {delay_new} sec is between offload and unoffload thresholds or no dege service, continue with next lambda value")
            cost_v[t,lambda_i,a] = cost_v[t,lambda_i-1,a]
            cost_v_edge[t,lambda_i,a] = cost_v_edge[t,lambda_i-1,a]
            cost_v_cloud[t,lambda_i,a] = cost_v_cloud[t,lambda_i-1,a]
            delay_v[t,lambda_i,a] = delay_v[t,lambda_i-1,a]
            delta_cost_v[t,lambda_i,a] = delta_cost_v[t,lambda_i-1,a]
            p_time_v[t,lambda_i,a] = 0
            rhoce_v[t,lambda_i,a] = rhoce_v[t,lambda_i-1,a]
            edge_ms_v[t,lambda_i,a] = edge_ms_v[t,lambda_i-1,a]
            opt=False
        if opt:
            params = {
                'S_edge_b': S_b_new[M:],
                'Acpu': Acpu_new.copy(),
                'Amem': Amem_new.copy(),
                'Qcpu': Qcpu.copy(),
                'Qmem': Qmem.copy(),
                'Fcm': Fcm.copy(),
                'M': M,
                'lambd': lambda_val,
                'Rs': Rs.copy(),
                'Di': Di.copy(),
                'delay_decrease_target': delay_decrease_target,
                'delay_increase_target': delay_increase_target,
                'RTT': RTT,
                'Ne': Ne,
                'Cost_cpu_edge': Cost_cpu_edge,
                'Cost_mem_edge': Cost_mem_edge,
                'Cost_cpu_cloud': Cost_cpu_cloud,
                'Cost_mem_cloud': Cost_mem_cloud,
                'locked': None,
                'dependency_paths_b': None
            }
            tic = time.time()
            result = sbmp_o(params)[1] if mode == 'offload' else sbmp_u(params)[1]
            # if result['delay'] < unoffload_threshold and mode == 'offload':
            #     params['S_edge_b'] = result['S_edge_b']
            #     params['Amem'] = result['Amem']
            #     params['Acpu'] = result['Acpu']
            #     params['delay_increase_target'] = target_delay - result['delay']
            #     result = unoffload(params)[1]
            toc = time.time()
            print(f'processing time {alg_type[a]} {(toc-tic)} sec')
            if mode == 'offload':
                print(f"Result {alg_type[a]} for {mode} \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, rhoce: {result['rhoce']}")
            else:
                print(f"Result {alg_type[a]} for {mode} \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay : {result['delay']}, rhoce: {result['rhoce']}")
            cost_v[t,lambda_i,a] = result['Cost']
            delay_v[t,lambda_i,a] = result['delay']
            delta_cost_v[t,lambda_i,a] = delta_cost_v[t,lambda_i,a] = result['Cost'] - Cost_void
            cost_v_edge[t,lambda_i,a] = result['Cost_edge']
            cost_v_cloud[t,lambda_i,a] = result['Cost_cloud']
            p_time_v[t,lambda_i,a] = toc-tic
            rhoce_v[t,lambda_i,a] = result['rhoce']
            edge_ms_v[t,lambda_i,a] = np.sum(result['S_edge_b'])-1
            S_b_old_dict[a][M:]=result['S_edge_b']
        
        # MFU ##
        a+=1
        alg_type[a] = "MFU"
        S_b_new = S_b_old_dict[a].copy()
        Acpu_new = np.zeros(2*M)
        Amem_new = np.zeros(2*M)
        Fci_new = np.matrix(buildFi(S_b_new, Fcm, M))
        Nci_new = computeN(Fci_new, M, 2)
        computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_void,Amem_void,Nci_void)
        delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, np.tile(Rs,2), RTT, Ne, lambda_val, M, np.empty(0))[0] # Total delay of the temp state. It includes only network delays
        if delay_new > offload_threshold:
            delay_decrease_target = delay_new - target_delay
            delay_increase_target = 0
            mode = 'offload'
            opt=True
        elif delay_new < unoffload_threshold and np.sum(S_b_new[M:2*M-1])>0:
            delay_increase_target = target_delay - delay_new
            delay_decrease_target = 0
            mode = 'unoffload'
            opt=True
        else:
            print(f"Delay {delay_new} sec is between offload and unoffload thresholds or no edge service, continue with next lambda value")
            cost_v[t,lambda_i,a] = cost_v[t,lambda_i-1,a]
            cost_v_edge[t,lambda_i,a] = cost_v_edge[t,lambda_i-1,a]
            cost_v_cloud[t,lambda_i,a] = cost_v_cloud[t,lambda_i-1,a]
            delay_v[t,lambda_i,a] = delay_v[t,lambda_i-1,a]
            delta_cost_v[t,lambda_i,a] = delta_cost_v[t,lambda_i-1,a]
            p_time_v[t,lambda_i,a] = 0
            rhoce_v[t,lambda_i,a] = rhoce_v[t,lambda_i-1,a]
            edge_ms_v[t,lambda_i,a] = edge_ms_v[t,lambda_i-1,a]
            opt=False
        if opt:
            params = {
                'S_edge_b': S_b_new[M:],
                'Acpu': Acpu_new.copy(),
                'Amem': Amem_new.copy(),
                'Fcm': Fcm.copy(),
                'Qcpu': Qcpu.copy(),
                'Qmem': Qmem.copy(),
                'M': M,
                'lambd': lambda_val,
                'Rs': Rs.copy(),
                'Di': Di.copy(),
                'delay_decrease_target': delay_decrease_target,
                'delay_increase_target': delay_increase_target,
                'RTT': RTT,
                'Ne': Ne,
                'Cost_cpu_edge': Cost_cpu_edge,
                'Cost_mem_edge': Cost_mem_edge,
                'Cost_cpu_cloud': Cost_cpu_cloud,
                'Cost_mem_cloud': Cost_mem_cloud,
                'mode': mode
            }
            tic = time.time()
            result = mfu(params)
            toc = time.time()
            print(f'processing time {alg_type[a]} {(toc-tic)} sec')
            if mode == 'offload':
                print(f"Result {alg_type[a]} for {mode} \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, rhoce: {result['rhoce']}")
            else:
                print(f"Result {alg_type[a]} for {mode} \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay : {result['delay']}, rhoce: {result['rhoce']}")
            cost_v[t,lambda_i,a] = result['Cost']
            delay_v[t,lambda_i,a] = result['delay']
            delta_cost_v[t,lambda_i,a] = result['Cost'] - Cost_void
            cost_v_edge[t,lambda_i,a] = result['Cost_edge']
            cost_v_cloud[t,lambda_i,a] = result['Cost_cloud']
            p_time_v[t,lambda_i,a] = toc-tic
            rhoce_v[t,lambda_i,a] = result['rhoce']
            edge_ms_v[t,lambda_i,a] = np.sum(result['S_edge_b'])-1
            S_b_old_dict[a][M:]=result['S_edge_b']
        
    # Matlab save
    mdic = {"best_cost_v": cost_v, "best_cost_v_edge": cost_v_edge, "best_cost_v_cloud": cost_v_cloud, "best_delay_v": delay_v, "best_delta_cost_v": delta_cost_v, "p_time_v": p_time_v, 'rhoce_v': rhoce_v, 'edge_ms_v': edge_ms_v, 'alg_type': alg_type, 'lambda_range': lambda_range}
    savemat("resd1.mat", mdic)


