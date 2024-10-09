# simultion of offloading algorithms varing the dependency graph edges and number of microservices

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from old.EPAMP_offload_caching import offload
from MFU_heuristic import mfu_heuristic
from IA_heuristic import IA_heuristic
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeNc import computeNc
from scipy.io import savemat
from buildFci import buildFci
from numpy import inf
import time
import utils
import random
import logging

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

logging.basicConfig(stream=sys.stdout, level='ERROR',format='%(asctime)s GMA %(levelname)s %(message)s')

M_max = 211
max_algotithms = 10

trials = 50
seed = 150273
np.random.seed(seed)
random.seed(seed)
RTT = 0.06    # RTT edge-cloud
delay_decrease_target = 0.03    # requested delay reduction
lambda_val = 50     # request per second
Ne =1e9    # bitrate cloud-edge

graph_algorithm = 'barabasi' # 'random' or 'barabasi
barabasi=dict()
barabasi['m'] = 1
barabasi['power'] = 0.05
barabasi['zero_appeal'] = 0.01
random=dict()
random['n_parents'] = 3

Fcm_range_min = 0.1 # min value of microservice call frequency 
Fcm_range_max = 0.5 # max value of microservice call frequency 
Acpu_range_min = 0.5  # min value of actual CPU consumption per instance-set
Acpu_range_max = 8 # max value of actual CPU consumption per instance-set
Rs_range_min = 200000 # min value of response size in bytes
Rs_range_max = 2000000   # max of response size in bytes
Di_range_min = 0.01 # min value of internal delay (sec)
Di_range_max = 0.08 # max value of internal delay (Sec)

Cost_cpu_edge = 1 # cost of CPU at the edge
Cost_mem_edge = 1 # cost of memory at the edge

show_graph = False
show_plot = False

best_cost_v = np.empty((trials,int(M_max/10),max_algotithms)) # vector of costs obtained by different algorithms 
best_delta_v = np.empty((trials,int(M_max/10),max_algotithms)) # vector of delta obtained by different algorithms  
p_time_v = np.empty((trials,int(M_max/10),max_algotithms)) # vector of processing time obtained by different algorithms

Mi=-1
for M in range(11,M_max,10):
    Mi+=1   # index of the number of microservices
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
    S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
    S_b[M-1] = 0  # User is not in the cloud
    Qcpu = np.ones(2*M) # CPU quota per ms
    Qmem = np.ones(2*M) # memory quota per ms
    Di = np.random.uniform(Di_range_min,Di_range_max,M)
    Di[M-1] = 0 # user has no internal delay
    Di = np.tile(Di, 2) # double the size of Di for the cloud and edge instances
    barabasi['n'] = M-1
    
    for k in range(trials):
        print(f'\n\ntrial {k}')  
        Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
        Rs[M-1]=0 # user has no response size
        
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

        # set random values for CPU and memory requests in case of cloud only deployment
        Acpu_void = np.random.uniform(Acpu_range_min,Acpu_range_max,size=M)
        Acpu_void[M-1]=0   # user has no CPU request
        Amem_void = np.zeros(M)
        Acpu_void = np.append(Acpu_void, np.zeros(M))
        Amem_void = np.append(Amem_void, np.zeros(M))
        S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
        S_b_void[M-1] = 0  # User is not in the cloud
        S_b_void[2*M-1] = 1  # User is in the cloud
        Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
        Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
        
        # compute Acpu and Amem for the current state
        # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
        Fci = np.matrix(buildFci(S_b, Fcm, M))    # instance-set call frequency matrix of the current state
        Nci = computeNc(Fci, M, 2)    # number of instance call per user request of the current state
        Acpu = Acpu_void.copy()
        Amem = Amem_void.copy()
        utils.computeResourceShift(Acpu, Amem, Nci, Acpu_void, Amem_void, Nci_void) # compute the resource shift from void state to the current S_b state

        # if k!=3:
        #     continue
    
        if show_graph:
            G = nx.DiGraph(Fcm)
            nx.draw(G,with_labels=True)
            plt.show()
        
        alg_type = [""] * max_algotithms # vector of strings describing algorithms used in a trial
        a=-1
        
        ## E_PAMP ##
        # a+=1
        # alg_type[a] = "E_PAMP no upgrade limit"
        # params = {
        #     'S_edge_b': S_edge_b.copy(),
        #     'Acpu': Acpu.copy(),
        #     'Amem': Amem.copy(),
        #     'Qcpu': Qcpu.copy(),
        #     'Qmem': Qmem.copy(),
        #     'Fcm': Fcm.copy(),
        #     'M': M,
        #     'lambd': lambda_val,
        #     'Rs': Rs,
        #     'Di': Di,
        #     'delay_decrease_target': delay_decrease_target,
        #     'RTT': RTT,
        #     'Ne': Ne,
        #     'Cost_cpu_edge': Cost_cpu_edge,
        #     'Cost_mem_edge': Cost_mem_edge,
        #     'locked': None,
        #     'dependency_paths_b': None,
        #     'u_limit': M
        # }
        # tic = time.time()
        # result = offload(params)[1]
        # toc = time.time()
        # print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        # print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        # best_cost_v[k,Mi,a] = result['Cost']
        # best_delta_v[k,Mi,a] = result['delay_decrease']
        # p_time_v[k,Mi,a] = toc-tic
        
        a+=1
        alg_type[a] = "E_PAMP with upgrade limit 2"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Acpu': Acpu.copy(),
            'Amem': Amem.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'Fcm': Fcm.copy(),
            'M': M,
            'lambd': lambda_val,
            'Rs': Rs,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'Ne': Ne,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'locked': None,
            'dependency_paths_b': None,
            'u_limit': 2
        }
        tic = time.time()
        result = offload(params)[1]
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        best_cost_v[k,Mi,a] = result['Cost']
        best_delta_v[k,Mi,a] = result['delay_decrease']
        p_time_v[k,Mi,a] = toc-tic

        # a+=1
        # alg_type[a] = "E_PAMP with upgrade limit 1"
        # params = {
        #     'S_edge_b': S_edge_b.copy(),
        #     'Acpu': Acpu.copy(),
        #     'Amem': Amem.copy(),
        #     'Qcpu': Qcpu.copy(),
        #     'Qmem': Qmem.copy(),
        #     'Fcm': Fcm.copy(),
        #     'M': M,
        #     'lambd': lambda_val,
        #     'Rs': Rs,
        #     'Di': Di,
        #     'delay_decrease_target': delay_decrease_target,
        #     'RTT': RTT,
        #     'Ne': Ne,
        #     'Cost_cpu_edge': Cost_cpu_edge,
        #     'Cost_mem_edge': Cost_mem_edge,
        #     'locked': None,
        #     'dependency_paths_b': None,
        #     'u_limit': 1
        # }
        # tic = time.time()
        # result = offload(params)[1]
        # toc = time.time()
        # print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        # print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        # best_cost_v[k,Mi,a] = result['Cost']
        # best_delta_v[k,Mi,a] = result['delay_decrease']
        # p_time_v[k,Mi,a] = toc-tic
        
        
        ## MFU ##
        a+=1
        alg_type[a] = "MFU"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Acpu': Acpu.copy(),
            'Amem': Amem.copy(),
            'Fcm': Fcm.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'M': M,
            'lambd': lambda_val,
            'Rs': Rs,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'Ne': Ne,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'mode': 'offload'
        }
        tic = time.time()
        result = mfu_heuristic(params)
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        best_cost_v[k,Mi,a] = result['Cost']
        best_delta_v[k,Mi,a] = result['delay_decrease']
        p_time_v[k,Mi,a] = toc-tic
        
        ## IA ##
        a+=1
        alg_type[a] = "IA"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Acpu': Acpu.copy(),
            'Amem': Amem.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'Fcm': Fcm.copy(),
            'M': M,
            'lambd': lambda_val,
            'Rs': Rs,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'Ne': Ne,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge
        }
        tic = time.time()
        result = IA_heuristic(params)
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        best_cost_v[k,Mi,a] = result['Cost']
        best_delta_v[k,Mi,a] = result['delay_decrease']
        p_time_v[k,Mi,a] = toc-tic
            
    if show_plot:
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x', '+']
        for i in range(a+1):
            line, = plt.plot(best_cost_v[:,Mi,0], best_cost_v[:,Mi,i], label=alg_type[i], linestyle='none', marker=markers[i])
        plt.ylabel('cost')
        plt.xlabel(f'cost of {alg_type[0]}')
        plt.legend()
        plt.show()

    # Matlab save
    mdic = {"best_cost_v": best_cost_v, "best_delta_v": best_delta_v, "p_time_v": p_time_v}
    savemat("res3.mat", mdic)
