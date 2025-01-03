# simultion of offloading algorithms varing the dependency graph edges and number of microservices

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(f'{parent_dir}/utils')
sys.path.append(f'{parent_dir}/strategies')

from SBMP_offload import sbmp_o
from SBMP_unoffload import sbmp_u
from MFU import mfu
from IA import IA_heuristic
from igraph import *
from utils import buildFi, computeN, computeDTot, computeCost, computeResourceShift
from numpy import inf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import logging
import graph_gen

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

logging.basicConfig(stream=sys.stdout, level='ERROR',format='%(asctime)s GMA %(levelname)s %(message)s')

M = 121 # number of microservices
max_algotithms = 10

trials = 50
seed = 150273
np.random.seed(seed)
random.seed(seed)
RTT = 0.08    # RTT edge-cloud
max_target_delay = 0.181   # max target delay  
min_target_delay = 0.06   # min target delay 
app_delay_no_net = 0.05 # delay of the application without network
#delay_decrease_target = 0.03    # requested delay reduction
lambda_val = 50    # request per second
B =1e9    # bitrate cloud-edge

graph_algorithm = 'barabasi-pareto' # 'random' or 'barabasi' or
barabasi=dict()
barabasi['m'] = 1
barabasi['power'] = 0.9
barabasi['zero_appeal'] = 3.125
barabasi['pareto_shape'] = 10
random=dict()
random['n_parents'] = 3

Fm_range_min = 0.1 # min value of microservice call frequency 
Fm_range_max = 0.5 # max value of microservice call frequency 
u_scale = 10 # scaling factor for CPU usage per req/sec
Ucpu_range_min = 5  # min value of CPU usage per cloud/edge instance set per u_scale req/s
Ucpu_range_max = 40 # max value of CPU usage per cloud/edge instance set per u_scale req/s
L_range_min = 500000 # min value of response size in bytes
L_range_max = 3500000   # max of response size in bytes
Di_range_min = 0.005 # min value of internal delay (sec)
Di_range_max = 0.01 # max value of internal delay (sec)

Cost_cpu_edge = 0.056/2 # cost of CPU at the edge per hour
Cost_mem_edge = 0.056/4 # cost of 1 GB memory at the edge per hour
Cost_cpu_cloud = 0.0416/2 # cost of CPU at the edge per hour
Cost_mem_cloud = 0.0416/4 # cost of 1 GB memory at the edge per hour
Cost_network = 0.02 # cost of network traffic per GB

show_graph = False
show_plot = False

x_samples = np.ceil((max_target_delay - min_target_delay)/0.01)
cost_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of costs obtained by different algorithms 
cost_v_edge = np.empty((trials,int(x_samples),max_algotithms)) # vector of edge costs obtained by different algorithms 
cost_v_cloud = np.empty((trials,int(x_samples),max_algotithms)) # vector of cloud costs obtained by different algorithms
cost_v_traffic = np.empty((trials,int(x_samples),max_algotithms)) # vector of network costs obtained by different algorithms
delay_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of delta obtained by different algorithms 
delta_cost_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of delta obtained by different algorithms 
p_time_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of processing time obtained by different algorithms
edge_ms_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of number of edge microservice obtained by different algorithms
rhoce_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of edge-cloud network utilization obtained by different algorithms
lambda_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of lambda used by different algorithms
target_delay_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of target delay used by different algorithms
n_microservices_v = np.empty((trials,int(x_samples),max_algotithms)) # vector of number of microservices used by different algorithms
  
for k in range(trials):
    print(f'\n\ntrial {k}')
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
    S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
    S_b[M-1] = 0  # User is not in the cloud
    Qcpu = np.ones(2*M) # CPU quota, not considered CPU request as cost model
    Qmem = np.zeros(2*M)*2 # memory quota, not considered memory request as cost model
    Di = np.random.uniform(Di_range_min,Di_range_max,M)
    Di[M-1] = 0 # user has no internal delay
    Di = np.tile(Di, 2)    
    barabasi['n'] = M-1  
    L = np.random.randint(L_range_min,L_range_max,M)  # random response size bytes
    L[0] = 2e6 # ingress microservice has response size of 2MB
    L[M-1]=0 # user has no response size
    #lambda_val = B*0.8/(L[0]*8) # lambda is set to 80% of the edge-cloud network capacity
    
    # build dependency graph
    Fm = np.zeros([M,M])   # microservice call frequency matrix
    if graph_algorithm == 'barabasi':
        g = Graph.Barabasi(n=barabasi['n'],m=barabasi['m'],power=barabasi['power'],zero_appeal=barabasi['zero_appeal'], directed=True)
        edges_reversal(g)
        Fm[:M-1,:M-1] = np.matrix(g.get_adjacency()) 
    elif graph_algorithm == 'random':
        n_parents = 3
        for i in range(1,M-1):
            n_parent=np.random.randint(1,random['n_parents'])
            for j in range(n_parent):
                a = np.random.randint(i)
                Fm[a,i]=1
    elif graph_algorithm=='barabasi-pareto':
        g = graph_gen.barabasi_albert_with_pareto(n=barabasi['n'], pshape=barabasi['pareto_shape'], power=barabasi['power'], zero_appeal=barabasi['zero_appeal'])
        Fm[:M-1,:M-1] = nx.adjacency_matrix(g).toarray()  
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fm[i,j]=np.random.uniform(Fm_range_min,Fm_range_max) if Fm[i,j]>0 else 0
    Fm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)

    # set random values for CPU and memory requests in case of cloud only deployment
    Ucpu_void = np.random.uniform(Ucpu_range_min,Ucpu_range_max,size=M) * lambda_val / u_scale
    Ucpu_void[M-1]=0   # user has no CPU request
    Umem_void = Ucpu_void * 2 # memory request is twice the CPU request (rule of thumb 1 CPU x 2GBs)
    Ucpu_void = np.append(Ucpu_void, np.zeros(M))
    Umem_void = np.append(Umem_void, np.zeros(M))
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fi_void = np.matrix(buildFi(S_b_void, Fm, M))    # instance-set call frequency matrix of the void state
    N_void = computeN(Fi_void, M, 2)    # number of instance call per user request of the void state
    
    # compute Ucpu and Umem for the current state
    # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
    Fi = np.matrix(buildFi(S_b, Fm, M))    # instance-set call frequency matrix of the current state
    N = computeN(Fi, M, 2)    # number of instance call per user request of the current state
    Ucpu = Ucpu_void.copy()
    Umem = Umem_void.copy()
    computeResourceShift(Ucpu, Umem, N, Ucpu_void, Umem_void, N_void) # compute the resource shift from void state to the current S_b state
    delay_no_network,_,_,rhoce_no_network = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), 0, np.inf, lambda_val, M) # compute the delay of the void state without network delay, equals to full edge delay
    # Di rescaling
    scaling_factor = delay_no_network / app_delay_no_net
    Di = Di / scaling_factor
    delay_no_network,_,_,rhoce_no_network = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), 0, np.inf, lambda_val, M) # compute the delay of the void state without network delay, equals to full edge delay
    delay_void,_,_,rhoce_void = computeDTot(S_b_void, N_void, Fi_void, Di, np.tile(L, 2), RTT, B, lambda_val, M)
    Cost_sum_void, Cost_edge_void, Cost_cloud_void, Cost_traffic_ce_void = computeCost(Ucpu_void,Umem_void,Qcpu,Qmem,Cost_cpu_edge,Cost_mem_edge,Cost_cpu_cloud,Cost_mem_cloud, rhoce_void * B, Cost_network,HPA_cpu_th=None)
    if show_graph:
        G = nx.DiGraph(Fm)
        nx.draw(G,with_labels=True)
        plt.show()
    
    Ti=-1  
    for target_delay in np.arange(min_target_delay,max_target_delay,0.01):
        print(f'\n target_delay {target_delay}')
        delay_decrease_target = max(delay_void - target_delay,0)
        Ti+=1   # index of the delay target   
        alg_type = [""] * max_algotithms # vector of strings describing algorithms used in a trial
        
        a=-1
        ## SBMP ##
        a+=1
        alg_type[a] = "SBMP - Traces"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Ucpu': Ucpu.copy(),
            'Umem': Umem.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'Fm': Fm.copy(),
            'M': M,
            'lambd': lambda_val,
            'L': L,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'sgs-builder': 'sgs_builder_traces',
            'expanding-depth': 2,
            'max-sgs': 256,
            'max-traces': 2048,
        }
        tic = time.time()
        result = sbmp_o(params)[2]
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        cost_v[k,Ti,a] = result['Cost']
        cost_v_edge[k,Ti,a] = result['Cost_edge']
        cost_v_cloud[k,Ti,a] = result['Cost_cloud']
        delay_v[k,Ti,a] = result['delay']
        rhoce_v[k,Ti,a] = result['rhoce']
        delta_cost_v[k,Ti,a] = result['cost_increase']
        cost_v_traffic[k,Ti,a] = result['Cost_traffic']
        p_time_v[k,Ti,a] = toc-tic
        edge_ms_v[k,Ti,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,Ti,a] = lambda_val
        target_delay_v[k,Ti,a] = target_delay
        n_microservices_v[k,Ti,a] = M

        ## MFU ##
        a+=1
        alg_type[a] = "MFU"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Ucpu': Ucpu.copy(),
            'Umem': Umem.copy(),
            'Fm': Fm.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'M': M,
            'lambd': lambda_val,
            'L': L,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'mode': 'offload'
        }
        tic = time.time()
        result = mfu(params)[2]
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        cost_v[k,Ti,a] = result['Cost']
        cost_v_edge[k,Ti,a] = result['Cost_edge']
        cost_v_cloud[k,Ti,a] = result['Cost_cloud']
        delay_v[k,Ti,a] = result['delay']
        rhoce_v[k,Ti,a] = result['rhoce']
        delta_cost_v[k,Ti,a] = result['cost_increase']
        cost_v_traffic[k,Ti,a] = result['Cost_traffic']
        p_time_v[k,Ti,a] = toc-tic
        edge_ms_v[k,Ti,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,Ti,a] = lambda_val
        target_delay_v[k,Ti,a] = target_delay
        n_microservices_v[k,Ti,a] = M
        
        ## SBMP ##
        a+=1
        alg_type[a] = "SBMP - SPA (PAMP)"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Ucpu': Ucpu.copy(),
            'Umem': Umem.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'Fm': Fm.copy(),
            'M': M,
            'lambd': lambda_val,
            'L': L,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'sgs-builder': 'sgs_builder_COPAMP',
            'expanding-depth': 2
        }
        tic = time.time()
        result = sbmp_o(params)[2]
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        cost_v[k,Ti,a] = result['Cost']
        cost_v_edge[k,Ti,a] = result['Cost_edge']
        cost_v_cloud[k,Ti,a] = result['Cost_cloud']
        delay_v[k,Ti,a] = result['delay']
        rhoce_v[k,Ti,a] = result['rhoce']
        delta_cost_v[k,Ti,a] = result['cost_increase']
        cost_v_traffic[k,Ti,a] = result['Cost_traffic']
        p_time_v[k,Ti,a] = toc-tic
        edge_ms_v[k,Ti,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,Ti,a] = lambda_val
        target_delay_v[k,Ti,a] = target_delay
        n_microservices_v[k,Ti,a] = M


        # IA ##
        a+=1
        alg_type[a] = "IA"
        params = {
            'S_edge_b': S_edge_b.copy(),
            'Ucpu': Ucpu.copy(),
            'Umem': Umem.copy(),
            'Qcpu': Qcpu.copy(),
            'Qmem': Qmem.copy(),
            'Fm': Fm.copy(),
            'M': M,
            'lambd': lambda_val,
            'L': L,
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
        }
        tic = time.time()
        result = IA_heuristic(params)[2]
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
        cost_v[k,Ti,a] = result['Cost']
        cost_v_edge[k,Ti,a] = result['Cost_edge']
        cost_v_cloud[k,Ti,a] = result['Cost_cloud']
        delay_v[k,Ti,a] = result['delay']
        rhoce_v[k,Ti,a] = result['rhoce']
        cost_v_traffic[k,Ti,a] = result['Cost_traffic']
        delta_cost_v[k,Ti,a] = result['cost_increase']
        p_time_v[k,Ti,a] = toc-tic
        edge_ms_v[k,Ti,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,Ti,a] = lambda_val
        target_delay_v[k,Ti,a] = target_delay
        n_microservices_v[k,Ti,a] = M
    
    if show_plot:
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x', '+']
        for i in range(a+1):
            line, = plt.plot(cost_v[:,Ti,0], cost_v[:,Ti,i], label=alg_type[i], linestyle='none', marker=markers[i])
        plt.ylabel('cost')
        plt.xlabel(f'cost of {alg_type[0]}')
        plt.legend()
        plt.show()

    # Matlab save
mdic = {"cost_v": cost_v, "cost_v_edge": cost_v_edge, "cost_v_cloud": cost_v_cloud, "delay_v": delay_v, "delta_cost_v": delta_cost_v, "p_time_v": p_time_v, "edge_ms_v": edge_ms_v, "rhoce_v": rhoce_v, "cost_v_traffic": cost_v_traffic, "lambda_v": lambda_v, "target_delay_v": target_delay_v, "n_microservices_v": n_microservices_v}
savemat(f"res1_const_size_pareto{barabasi['pareto_shape']}.mat", mdic)
