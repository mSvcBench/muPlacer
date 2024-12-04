# simultion of offloading algorithms varing the dependency graph edges and number of microservices

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from SBMP_offload import sbmp_o
from SBMP_unoffload import sbmp_u
from MFU import mfu
from IA import IA_heuristic
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeN import computeN
from scipy.io import savemat
from buildFi import buildFi
from numpy import inf
from computeDTot import computeDTot
import time
import utils
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
offload_threshold = 0.120 # threshold to offload in sec
unoffload_threshold = 0.08 # threshold to unoffload in sec
target_delay = unoffload_threshold + (offload_threshold-unoffload_threshold)/2.0 # target user delay (sec)
app_delay_no_net = 0.05 # delay of the application without network
#delay_decrease_target = 0.03    # requested delay reduction
lambda_min = 40   # min user request rate (req/s)
lambda_max = 500   # max user request rate (req/s)
lambda_step = 40   # user request rate step (req/s)
lambda_range = list(range(lambda_min, lambda_max+lambda_step, lambda_step))  # user request rates (req/s)
lambda_range = lambda_range + list(reversed(range(lambda_min, lambda_max+lambda_step, lambda_step)))  # user request rates (req/s)

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

x_samples = len(lambda_range)
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
  

# common parameters of scenarios
S_edge_b = np.zeros(M)  # initial state. 
S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
S_b_void = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
S_b_void[M-1] = 0  # User is not in the cloud
Qcpu = np.ones(2*M) # CPU quota, not considered CPU request as cost model
Qmem = np.zeros(2*M)*2 # memory quota, not considered memory request as cost model

# per-trial dynamic parameters of scenarios
scenario = dict()
scenario['Fm'] = dict()
scenario['Ucpu_void'] = dict()
scenario['Umem_void'] = dict()
scenario['Di'] = dict()
scenario['L'] = dict()
scenario['Cost_sum_void'] = dict()

for k in range(trials):
    print(f'Processing trial parameters {k}')
    Di = np.random.uniform(Di_range_min,Di_range_max,M)
    Di[M-1] = 0 # user has no internal delay
    Di = np.tile(Di, 2)    
    barabasi['n'] = M-1  
    L = np.random.randint(L_range_min,L_range_max,M)  # random response size bytes
    L[0] = 2e6 # ingress microservice has response size of 2MB
    L[M-1]=0 # user has no response size
    
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
    
    Ucpu_void = np.random.uniform(Ucpu_range_min,Ucpu_range_max,size=M) * lambda_min / u_scale    # set random values for CPU usage per req/sec
    Ucpu_void[M-1]=0   # user has no CPU request
    Ucpu_void = np.append(Ucpu_void, np.zeros(M))
    Umem_void = Ucpu_void * 2 # memory request is twice the CPU request (rule of thumb 1 CPU x 2GBs)
    
    # store scenario parameters
    scenario['Fm'][k] = Fm
    scenario['Ucpu_void'][k] = Ucpu_void    
    scenario['Umem_void'][k] = Umem_void
    scenario['Di'][k] = Di
    scenario['L'][k] = L
    

alg_type = [""] * max_algotithms # vector of strings describing algorithms used in a trial
a=-1

a+=1
for k in range(trials):
    ## SBMP ##
    alg_type[a] = "SBMP - Traces"
    S_b = S_b_void.copy()
    Ucpu = scenario['Ucpu_void'][k].copy()
    Umem = scenario['Umem_void'][k].copy()
    L = scenario['L'][k]
    Di = scenario['Di'][k]
    Fm = scenario['Fm'][k]
    Fi = np.matrix(buildFi(S_b, Fm, M))
    N = computeN(Fi, M, 2)

    traces_b = utils.sgs_builder_traces_full(M,2048,Fm)
    
    for lambda_i,lambda_val in enumerate(lambda_range):

        delay,_,_,rhoce = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), RTT, B, lambda_val, M) # compute the delay of the void state without network delay, equals to full edge delay
        if lambda_i>0:
            Ucpu = Ucpu * lambda_val / lambda_range[lambda_i-1]
            Umem = 2 * Ucpu
        
        # compute cost of cloud only deployment
        Ucpu_void = np.zeros(2*M)
        Umem_void = np.zeros(2*M)
        Ucpu_void[:M] = Ucpu[:M]+Ucpu[M:]
        Umem_void[:M] = Umem[:M]+Umem[M:]
        B_void = L[0]*8*lambda_val
        Cost_void = utils.computeCost(Ucpu_void, Umem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, B_void, Cost_network)[0]  
        
        if delay > offload_threshold:
            delay_decrease_target = delay - target_delay
            delay_increase_target = 0
            mode = 'offload'
        elif delay < unoffload_threshold and np.sum(S_b[M:2*M-1]):
            delay_decrease_target = 0
            delay_increase_target = target_delay - delay
            mode = 'unoffload'
        else:
            mode = 'none'
            
        params = {
            'S_edge_b': S_b[M:].copy(),
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
            'delay_increase_target': delay_increase_target,
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
            'traces-b': traces_b
        }
        tic = time.time()
        if mode == 'offload':
            result = sbmp_o(params)[2]
            print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        elif mode == 'unoffload':
            result = sbmp_u(params)[2]
            print(f"Result {alg_type[a]} for unoffload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay increase: {result['delay_increase']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        else:
            Cost_sum, Cost_edge, Cost_cloud, Cost_traffic_ce = utils.computeCost(Ucpu, Umem, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce * B, Cost_network)
            result['delay'] = delay
            result['Ucpu'] = Ucpu
            result['Umem'] = Umem
            result['Cost'] = Cost_sum
            result['Cost_edge'] = Cost_edge
            result['Cost_cloud'] = Cost_cloud
            result['Cost_traffic'] = Cost_traffic_ce
            result['rhoce'] = rhoce

            print(f"Result {alg_type[a]} for none \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        
        cost_v[k,lambda_i,a] = result['Cost']
        cost_v_edge[k,lambda_i,a] = result['Cost_edge']
        cost_v_cloud[k,lambda_i,a] = result['Cost_cloud']
        delay_v[k,lambda_i,a] = result['delay']
        rhoce_v[k,lambda_i,a] = result['rhoce']
        delta_cost_v[k,lambda_i,a] = result['Cost']-Cost_void
        cost_v_traffic[k,lambda_i,a] = result['Cost_traffic']
        p_time_v[k,lambda_i,a] = toc-tic
        edge_ms_v[k,lambda_i,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,lambda_i,a] = lambda_val
        target_delay_v[k,lambda_i,a] = target_delay
        n_microservices_v[k,lambda_i,a] = M

        S_b[M:] = result['S_edge_b']
        Fi = np.matrix(buildFi(S_b, Fm, M))
        N = computeN(Fi, M, 2)
        Ucpu = result['Ucpu']
        Umem = result['Umem']

a+=1
for k in range(trials):
    ## MFU ##
    alg_type[a] = "MFU"
    S_b = S_b_void.copy()
    Ucpu = scenario['Ucpu_void'][k].copy()
    Umem = scenario['Umem_void'][k].copy()
    L = scenario['L'][k]
    Di = scenario['Di'][k]
    Fm = scenario['Fm'][k]
    Fi = np.matrix(buildFi(S_b, Fm, M))
    N = computeN(Fi, M, 2)
    
    for lambda_i,lambda_val in enumerate(lambda_range):
        
        delay,_,_,rhoce = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), RTT, B, lambda_val, M)
        if lambda_i>0:
            Ucpu = Ucpu * lambda_val / lambda_range[lambda_i-1]
            Umem = 2 * Ucpu
        
        # compute cost of cloud only deployment
        Ucpu_void = np.zeros(2*M)
        Umem_void = np.zeros(2*M)
        Ucpu_void[:M] = Ucpu[:M]+Ucpu[M:]
        Umem_void[:M] = Umem[:M]+Umem[M:]
        B_void = L[0]*8*lambda_val
        Cost_void = utils.computeCost(Ucpu_void, Umem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, B_void, Cost_network)[0]  

        if delay > offload_threshold:
            delay_decrease_target = delay - target_delay
            delay_increase_target = 0
            mode = 'offload'
        elif delay < unoffload_threshold and np.sum(S_b[M:2*M-1]):
            delay_decrease_target = 0
            delay_increase_target = target_delay - delay
            mode = 'unoffload'
        else:
            mode = 'none'
            
        params = {
            'S_edge_b': S_b[M:].copy(),
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
            'delay_increase_target': delay_increase_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'mode': mode
        }
        tic = time.time()
        if mode == 'offload':
            result = mfu(params)[2]
            print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        elif mode == 'unoffload':
            result = mfu(params)[2]
            print(f"Result {alg_type[a]} for unoffload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, delay increase: {result['delay_increase']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        else:
            Cost_sum, Cost_edge, Cost_cloud, Cost_traffic_ce = utils.computeCost(Ucpu, Umem, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce * B, Cost_network)
            result['delay'] = delay
            result['Ucpu'] = Ucpu
            result['Umem'] = Umem
            result['Cost'] = Cost_sum
            result['Cost_edge'] = Cost_edge
            result['Cost_cloud'] = Cost_cloud
            result['Cost_traffic'] = Cost_traffic_ce
            result['rhoce'] = rhoce

            print(f"Result {alg_type[a]} for none \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
        
        cost_v[k,lambda_i,a] = result['Cost']
        cost_v_edge[k,lambda_i,a] = result['Cost_edge']
        cost_v_cloud[k,lambda_i,a] = result['Cost_cloud']
        delay_v[k,lambda_i,a] = result['delay']
        rhoce_v[k,lambda_i,a] = result['rhoce']
        delta_cost_v[k,lambda_i,a] = result['Cost']-Cost_void
        cost_v_traffic[k,lambda_i,a] = result['Cost_traffic']
        p_time_v[k,lambda_i,a] = toc-tic
        edge_ms_v[k,lambda_i,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,lambda_i,a] = lambda_val
        target_delay_v[k,lambda_i,a] = target_delay
        n_microservices_v[k,lambda_i,a] = M

        S_b[M:] = result['S_edge_b']
        Fi = np.matrix(buildFi(S_b, Fm, M))
        N = computeN(Fi, M, 2)
        Ucpu = result['Ucpu']
        Umem = result['Umem']

a+=1
for k in range(trials):
    ## CO-PAMP ##
    alg_type[a] = "CO-PAMP"
    S_b = S_b_void.copy()
    Ucpu = scenario['Ucpu_void'][k].copy()
    Umem = scenario['Umem_void'][k].copy()
    L = scenario['L'][k]
    Di = scenario['Di'][k]
    Fm = scenario['Fm'][k]
    Fi = np.matrix(buildFi(S_b, Fm, M))
    N = computeN(Fi, M, 2)

    traces_b = None
    
    for lambda_i,lambda_val in enumerate(lambda_range):  

        delay,_,_,rhoce = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), RTT, B, lambda_val, M) # compute the delay of the void state without network delay, equals to full edge delay
        if lambda_i>0:
            Ucpu = Ucpu * lambda_val / lambda_range[lambda_i-1]
            Umem = 2 * Ucpu
        
        # compute cost of cloud only deployment
        Ucpu_void = np.zeros(2*M)
        Umem_void = np.zeros(2*M)
        Ucpu_void[:M] = Ucpu[:M]+Ucpu[M:]
        Umem_void[:M] = Umem[:M]+Umem[M:]
        B_void = L[0]*8*lambda_val
        Cost_void = utils.computeCost(Ucpu_void, Umem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, B_void, Cost_network)[0]

        if delay > offload_threshold:
            delay_decrease_target = delay - target_delay
            delay_increase_target = 0
            mode = 'offload'
        elif delay < unoffload_threshold and np.sum(S_b[M:2*M-1]):
            delay_decrease_target = 0
            delay_increase_target = target_delay - delay
            mode = 'unoffload'
        else:
            mode = 'none'
            
        params = {
            'S_edge_b': S_b[M:].copy(),
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
            'delay_increase_target': delay_increase_target,
            'RTT': RTT,
            'B': B,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'Cost_cpu_cloud': Cost_cpu_cloud,
            'Cost_mem_cloud': Cost_mem_cloud,
            'Cost_network': Cost_network,
            'sgs-builder': 'sgs_builder_with_single_path_adding',
            'expanding-depth': 2,
            'max-sgs': 256,
            'max-traces': 2048,
            'traces-b': traces_b
        }
        tic = time.time()
        if mode == 'offload':
            result = sbmp_o(params)[2]
            print(f"Result {alg_type[a]} for offload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        elif mode == 'unoffload':
            result = sbmp_u(params)[2]
            print(f"Result {alg_type[a]} for unoffload \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        else:
            Cost_sum, Cost_edge, Cost_cloud, Cost_traffic_ce = utils.computeCost(Ucpu, Umem, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce * B, Cost_network)
            result['delay'] = delay
            result['Ucpu'] = Ucpu
            result['Umem'] = Umem
            result['Cost'] = Cost_sum
            result['Cost_edge'] = Cost_edge
            result['Cost_cloud'] = Cost_cloud
            result['Cost_traffic'] = Cost_traffic_ce
            result['rhoce'] = rhoce

            print(f"Result {alg_type[a]} for none \n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay: {result['delay']}, cost increase from void: {result['Cost']-Cost_void}, rhoce: {result['rhoce']}")
        
        toc = time.time()
        print(f'processing time {alg_type[a]} {(toc-tic)} sec')
                
        cost_v[k,lambda_i,a] = result['Cost']
        cost_v_edge[k,lambda_i,a] = result['Cost_edge']
        cost_v_cloud[k,lambda_i,a] = result['Cost_cloud']
        delay_v[k,lambda_i,a] = result['delay']
        rhoce_v[k,lambda_i,a] = result['rhoce']
        delta_cost_v[k,lambda_i,a] = result['Cost']-Cost_void
        cost_v_traffic[k,lambda_i,a] = result['Cost_traffic']
        p_time_v[k,lambda_i,a] = toc-tic
        edge_ms_v[k,lambda_i,a] = np.sum(result['S_edge_b'])-1
        lambda_v[k,lambda_i,a] = lambda_val
        target_delay_v[k,lambda_i,a] = target_delay
        n_microservices_v[k,lambda_i,a] = M

        S_b[M:] = result['S_edge_b']
        Fi = np.matrix(buildFi(S_b, Fm, M))
        N = computeN(Fi, M, 2)
        Ucpu = result['Ucpu']
        Umem = result['Umem']
    
    # if show_plot:
    #     markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x', '+']
    #     for i in range(a+1):
    #         line, = plt.plot(cost_v[k,:,0], cost_v[k,:,i], label=alg_type[i], linestyle='none', marker=markers[i])
    #     plt.ylabel('cost')
    #     plt.xlabel(f'cost of {alg_type[0]}')
    #     plt.legend()
    #     plt.show()

# Matlab save
mdic = {"cost_v": cost_v, "cost_v_edge": cost_v_edge, "cost_v_cloud": cost_v_cloud, "delay_v": delay_v, "delta_cost_v": delta_cost_v, "p_time_v": p_time_v, "edge_ms_v": edge_ms_v, "rhoce_v": rhoce_v, "cost_v_traffic": cost_v_traffic, "lambda_v": lambda_v, "target_delay_v": target_delay_v, "n_microservices_v": n_microservices_v}
savemat(f"res1_dyn_pareto{barabasi['pareto_shape']}.mat", mdic)
