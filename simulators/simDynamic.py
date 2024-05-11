import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

from offload import offload
from unoffload import unoffload
from mfu_heuristic import mfu_heuristic
from IA_heuristic import IA_heuristic
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeNc import computeNc
from computeDTot import computeDTot
from scipy.io import savemat
from buildFci import buildFci
from numpy import inf
import time

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

np.random.seed(150272)
res=np.array([])
trials = 30
RTT = 0.05    # RTT edge-cloud
M = 100 # n. microservices
Ne = 100e6    # bitrate cloud-edge
S_edge_b = np.zeros(M)  # initial state. 
S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
S_b[M-1] = 0  # User is not in the cloud
Cost_cpu_edge = 1 # cost of CPU at the edge
Cost_mem_edge = 1 # cost of memory at the edge
unoffload_margin = 0.1 # hysteresys margin for unoffloading

lambda_min = 500   # min user request rate (req/s)
lambda_max = 3000   # max user request rate (req/s)
lambda_step = 100   # user request rate step (req/s)
lambda_range = list(range(lambda_min, lambda_max+lambda_step, lambda_step))  # user request rates (req/s)
lambda_range = lambda_range + list(range(lambda_max-lambda_step, lambda_min-lambda_step, -lambda_step))  # user request rates (req/s)
target_delay = 0.25 # target user delay (sec)

graph_algorithm = 'random'

barabasi=dict()
barabasi['n'] = M-1
barabasi['m'] = 2
barabasi['power'] = 0.9
barabasi['zero_appeal'] = 0.9

random=dict()
random['n_parents'] = 3

Fcm_range_min = 0.1 # min value of microservice call frequency 
Fcm_range_max = 0.3 # max value of microservice call frequency 
Rcpu_quota = 0.5    # CPU quota
Rcpu_range_min = 1  # min value of requested CPU quota per instance-set
Rcpu_range_max = 32 # max value of requested CPU quota per instance-set
Rs_range_min = 1000 # min value of response size in bytes
Rs_range_max = 50000   # max of response size in bytes

Di_range_min = 0.01 # min value of internal delay (sec)
Di_range_max = 0.08 # max value of internal delay (Sec)

max_algotithms = 10

show_graph = False
show_plot = False


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
Rcpu_void = (np.random.randint(32,size=M)+1) * Rcpu_quota
Rcpu_void[M-1]=0   # user has no CPU request
Rmem_void = np.zeros(M)
Rcpu_void = np.append(Rcpu_void, np.zeros(M))
Rmem_void = np.append(Rmem_void, np.zeros(M))
S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
S_b_void[M-1] = 0  # User is not in the cloud
S_b_void[2*M-1] = 1  # User is in the cloud
Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state

# compute Rcpu and Rmem for the current state
# assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
Fci = np.matrix(buildFci(S_b, Fcm, M))    # instance-set call frequency matrix of the current state
Nci = computeNc(Fci, M, 2)    # number of instance call per user request of the current state
Rcpu = Rcpu_void.copy()
Rmem = Rmem_void.copy()
cloud_cpu_decrease = (1-Nci[:M]/Nci_void[:M]) * Rcpu_void[:M]   
cloud_mem_decrease = (1-Nci[:M]/Nci_void[:M]) * Rmem_void[:M]
cloud_cpu_decrease[np.isnan(cloud_cpu_decrease)] = 0
cloud_mem_decrease[np.isnan(cloud_mem_decrease)] = 0
cloud_cpu_decrease[cloud_cpu_decrease==-inf] = 0
cloud_mem_decrease[cloud_mem_decrease==-inf] = 0
Rcpu[M:] = Rcpu[M:] + cloud_cpu_decrease # edge cpu increase
Rmem[M:] = Rmem[M:] + cloud_mem_decrease # edge mem increase
Rcpu[:M] = Rcpu[:M] - cloud_cpu_decrease # cloud cpu decrease
Rmem[:M] = Rmem[:M] - cloud_mem_decrease # cloud mem decrease

# set random  internal delay
Di = np.random.uniform(Di_range_min,Di_range_max,M)
Di[M-1] = 0 # user has no internal delay
Di = np.tile(Di, 2)

if show_graph:
    G = nx.DiGraph(Fcm)
    nx.draw(G,with_labels=True)
    plt.show()



cost_v = np.empty((max_algotithms,len(lambda_range))) # vector of costs obtained by different algorithms during the load change
delay_v = np.empty((max_algotithms,len(lambda_range))) # vector of delay obtained by different algorithms during the load change
rhoce_v = np.empty((max_algotithms,len(lambda_range))) # vector of rhoce obtained by different algorithms during the load change
delay_old_v = np.empty((max_algotithms,len(lambda_range))) # vector of delay before action obtained by different algorithms during the load change
rhoce_old_v = np.empty((max_algotithms,len(lambda_range))) # vector of rhoch before action obtained by different algorithms during the load change
nmicros_v = np.empty((max_algotithms,len(lambda_range))) # vector of n. edge micros obtained by different algorithms during the load change
lambda_v = np.empty((max_algotithms,len(lambda_range))) # vector of lambdas used for the tests

alg_type = [""] * max_algotithms # vector of strings describing algorithms used for te tests

a=-1
## E_PAMP limit 2 ##
a+=1
k=-1
S_edge_b_new = S_b[M:].copy()
Rcpu_new = Rcpu.copy()
Rmem_new = Rmem.copy()
alg_type[a] = "E_PAMP with upgrade limit 2"
for lambda_val in lambda_range:
    k+=1
    print(f'\n lambda {lambda_val} req/s')
    S_b_old = np.concatenate((np.ones(M), S_edge_b_new)) 
    S_b_old[M-1] = 0
    Rcpu_old = Rcpu_new.copy()
    Rmem_old = Rmem_new.copy()
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))    # microservice call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)   # number of instance call per user request of the current state
    delay_old,di_old,dn_old,rhoce_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambda_val, M)   # total delay of the current state
    params = {
        'S_edge_b': S_b_old[M:],
        'Rcpu': Rcpu_old,
        'Rmem': Rmem_old,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 2,
        'no_caching': False
    }
    if delay_old > target_delay or k==0:
        delay_decrease_target = delay_old - target_delay
        params['delay_decrease_target'] = delay_decrease_target
        print(f"Offloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = offload(params)
    elif delay_old < (1-unoffload_margin)*target_delay:
        delay_increase_target = target_delay - delay_old
        params['delay_increase_target'] = delay_increase_target
        print(f"Unoffloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = unoffload(params)
    else:
        print(f"No action, current delay {delay_old} sec, target delay {target_delay} sec")
        cost_v[a,k] = cost_v[a,k-1]
        delay_v[a,k] = delay_old
        rhoce_v[a,k] = rhoce_old
        delay_old_v[a,k] = delay_old
        rhoce_old_v[a,k] = rhoce_old
        nmicros_v[a,k] = nmicros_v[a,k-1]
        lambda_v[a,k] = lambda_val
        continue
  
    # update state and values
    S_edge_b_new = result['S_edge_b']
    Rcpu_new = result['Rcpu']
    Rmem_new = result['Rmem']
    Fci_new = result['Fci']
    Nci_new = result['Nci']
    Cost_new = result['Cost']
    rhoce_new = result['rhoce']
    delay_new = result['delay']
    
    cost_v[a,k] = result['Cost']
    delay_v[a,k] = result['delay']
    delay_old_v[a,k] = delay_old
    rhoce_old_v[a,k] = rhoce_old
    rhoce_v[a,k] = result['rhoce']
    nmicros_v[a,k] = np.count_nonzero(result['S_edge_b'] > 0)
    lambda_v[a,k] = lambda_val
    print(f"edge instances: {np.argwhere(result['S_edge_b']==1).flatten()}")
    print(f"Cost: {cost_v[a, k]}, Delay: {delay_v[a, k]}, Rhoce: {rhoce_v[a, k]}, Nmicros: {nmicros_v[a, k]}, Lambda: {lambda_v[a, k]}")

## E_PAMP limit 1 ##
a+=1
k=-1
S_edge_b_new = S_b[M:].copy()
Rcpu_new = Rcpu.copy()
Rmem_new = Rmem.copy()
alg_type[a] = "E_PAMP with upgrade limit 1"
for lambda_val in lambda_range:
    k+=1
    print(f'\n lambda {lambda_val} req/s')
    S_b_old = np.concatenate((np.ones(M), S_edge_b_new)) 
    S_b_old[M-1] = 0
    Rcpu_old = Rcpu_new.copy()
    Rmem_old = Rmem_new.copy()
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))    # microservice call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)   # number of instance call per user request of the current state
    delay_old,di_old,dn_old,rhoce_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambda_val, M)   # total delay of the current state
    params = {
        'S_edge_b': S_b_old[M:],
        'Rcpu': Rcpu_old,
        'Rmem': Rmem_old,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 1,
        'no_caching': False
    }
    if delay_old > target_delay or k==0:
        delay_decrease_target = delay_old - target_delay
        params['delay_decrease_target'] = delay_decrease_target
        print(f"Offloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = offload(params)
    elif delay_old < (1-unoffload_margin)*target_delay:
        delay_increase_target = target_delay - delay_old
        params['delay_increase_target'] = delay_increase_target
        print(f"Unoffloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = unoffload(params)
    else:
        print(f"No action, current delay {delay_old} sec, target delay {target_delay} sec")
        cost_v[a,k] = cost_v[a,k-1]
        delay_v[a,k] = delay_old
        rhoce_v[a,k] = rhoce_old
        delay_old_v[a,k] = delay_old
        rhoce_old_v[a,k] = rhoce_old
        nmicros_v[a,k] = nmicros_v[a,k-1]
        lambda_v[a,k] = lambda_val
        continue
  
    # update state and values
    S_edge_b_new = result['S_edge_b']
    Rcpu_new = result['Rcpu']
    Rmem_new = result['Rmem']
    Fci_new = result['Fci']
    Nci_new = result['Nci']
    Cost_new = result['Cost']
    rhoce_new = result['rhoce']
    delay_new = result['delay']
    
    cost_v[a,k] = result['Cost']
    delay_v[a,k] = result['delay']
    delay_old_v[a,k] = delay_old
    rhoce_old_v[a,k] = rhoce_old
    rhoce_v[a,k] = result['rhoce']
    nmicros_v[a,k] = np.count_nonzero(result['S_edge_b'] > 0)
    lambda_v[a,k] = lambda_val
    print(f"edge instances: {np.argwhere(result['S_edge_b']==1).flatten()}")
    print(f"Cost: {cost_v[a, k]}, Delay: {delay_v[a, k]}, Rhoce: {rhoce_v[a, k]}, Nmicros: {nmicros_v[a, k]}, Lambda: {lambda_v[a, k]}")

## E_PAMP no limit  ##
a+=1
k=-1
S_edge_b_new = S_b[M:].copy()
Rcpu_new = Rcpu.copy()
Rmem_new = Rmem.copy()
alg_type[a] = "E_PAMP with upgrade limit 1"
for lambda_val in lambda_range:
    k+=1
    print(f'\n lambda {lambda_val} req/s')
    S_b_old = np.concatenate((np.ones(M), S_edge_b_new)) 
    S_b_old[M-1] = 0
    Rcpu_old = Rcpu_new.copy()
    Rmem_old = Rmem_new.copy()
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))    # microservice call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)   # number of instance call per user request of the current state
    delay_old,di_old,dn_old,rhoce_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambda_val, M)   # total delay of the current state
    params = {
        'S_edge_b': S_b_old[M:],
        'Rcpu': Rcpu_old,
        'Rmem': Rmem_old,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': M,
        'no_caching': False
    }
    if delay_old > target_delay or k==0:
        delay_decrease_target = delay_old - target_delay
        params['delay_decrease_target'] = delay_decrease_target
        print(f"Offloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = offload(params)
    elif delay_old < (1-unoffload_margin)*target_delay:
        delay_increase_target = target_delay - delay_old
        params['delay_increase_target'] = delay_increase_target
        print(f"Unoffloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = unoffload(params)
    else:
        print(f"No action, current delay {delay_old} sec, target delay {target_delay} sec")
        cost_v[a,k] = cost_v[a,k-1]
        delay_v[a,k] = delay_old
        rhoce_v[a,k] = rhoce_old
        delay_old_v[a,k] = delay_old
        rhoce_old_v[a,k] = rhoce_old
        nmicros_v[a,k] = nmicros_v[a,k-1]
        lambda_v[a,k] = lambda_val
        continue
  
    # update state and values
    S_edge_b_new = result['S_edge_b']
    Rcpu_new = result['Rcpu']
    Rmem_new = result['Rmem']
    Fci_new = result['Fci']
    Nci_new = result['Nci']
    Cost_new = result['Cost']
    rhoce_new = result['rhoce']
    delay_new = result['delay']
    
    cost_v[a,k] = result['Cost']
    delay_v[a,k] = result['delay']
    delay_old_v[a,k] = delay_old
    rhoce_old_v[a,k] = rhoce_old
    rhoce_v[a,k] = result['rhoce']
    nmicros_v[a,k] = np.count_nonzero(result['S_edge_b'] > 0)
    lambda_v[a,k] = lambda_val
    print(f"edge instances: {np.argwhere(result['S_edge_b']==1).flatten()}")
    print(f"Cost: {cost_v[a, k]}, Delay: {delay_v[a, k]}, Rhoce: {rhoce_v[a, k]}, Nmicros: {nmicros_v[a, k]}, Lambda: {lambda_v[a, k]}")

## MFU ##
a+=1
k=-1
S_edge_b_new = S_b[M:].copy()
Rcpu_new = Rcpu.copy()
Rmem_new = Rmem.copy()
alg_type[a] = "MFU"
for lambda_val in lambda_range:
    k+=1
    print(f'\n lambda {lambda_val} req/s')
    S_b_old = np.concatenate((np.ones(M), S_edge_b_new)) 
    S_b_old[M-1] = 0
    Rcpu_old = Rcpu_new.copy()
    Rmem_old = Rmem_new.copy()
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))    # microservice call frequency matrix
    Nci_old = computeNc(Fci_old, M, 2)   # number of instance call per user request of the current state
    delay_old,di_old,dn_old,rhoce_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambda_val, M)   # total delay of the current state
    params = {
        'S_edge_b': S_b_old[M:],
        'Rcpu': Rcpu_old,
        'Rmem': Rmem_old,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 2,
        'no_caching': False
    }
    if delay_old > target_delay or k==0:
        delay_decrease_target = delay_old - target_delay
        params['delay_decrease_target'] = delay_decrease_target
        print(f"Offloading, current delay {delay_old} sec, target delay {target_delay} sec")
        params['mode'] = 'offload'
        result = mfu_heuristic(params)
    elif delay_old < (1-unoffload_margin)*target_delay:
        delay_increase_target = target_delay - delay_old
        params['delay_increase_target'] = delay_increase_target
        params['mode'] = 'unoffload'
        print(f"Unoffloading, current delay {delay_old} sec, target delay {target_delay} sec")
        result = mfu_heuristic(params)
    else:
        print(f"No action, current delay {delay_old} sec, target delay {target_delay} sec")
        cost_v[a,k] = cost_v[a,k-1]
        delay_v[a,k] = delay_old
        rhoce_v[a,k] = rhoce_old
        delay_old_v[a,k] = delay_old
        rhoce_old_v[a,k] = rhoce_old
        nmicros_v[a,k] = nmicros_v[a,k-1]
        lambda_v[a,k] = lambda_val
        continue
  
    # update state and values
    S_edge_b_new = result['S_edge_b']
    Rcpu_new = result['Rcpu']
    Rmem_new = result['Rmem']
    Fci_new = result['Fci']
    Nci_new = result['Nci']
    Cost_new = result['Cost']
    rhoce_new = result['rhoce']
    delay_new = result['delay']
    
    cost_v[a,k] = result['Cost']
    delay_v[a,k] = result['delay']
    delay_old_v[a,k] = delay_old
    rhoce_old_v[a,k] = rhoce_old
    rhoce_v[a,k] = result['rhoce']
    nmicros_v[a,k] = np.count_nonzero(result['S_edge_b'] > 0)
    lambda_v[a,k] = lambda_val
    print(f"edge instances: {np.argwhere(result['S_edge_b']==1).flatten()}")
    print(f"Cost: {cost_v[a, k]}, Delay: {delay_v[a, k]}, Rhoce: {rhoce_v[a, k]}, Nmicros: {nmicros_v[a, k]}, Lambda: {lambda_v[a, k]}")

# Matlab save
mdic = {"cost_v": cost_v, "delay_v": delay_v, "delay_old_v": delay_old_v, "rhoce_v": rhoce_v, "rhoce_old_v": rhoce_old_v, "nmicros_v": nmicros_v,"lambda_v": lambda_v, "alg_type": alg_type}
savemat("res.mat", mdic)

plt.ion()
if show_plot:
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x', '+']
    plt.figure(0)
    for i in range(a+1):
        line, = plt.plot(lambda_v[a,:], cost_v[a,:], label=alg_type[i], marker=markers[i])
    plt.ylabel('cost')
    plt.xlabel(f'lambda req/s')
    plt.legend()
    plt.show()
    plt.pause(0.1)
    
    plt.figure(1)
    for i in range(a+1):
        line, = plt.plot(lambda_v[a,1:], delay_v[a,1:], label=alg_type[i], marker=markers[i], linestyle='none')
        line, = plt.plot(lambda_v[a,1:], delay_old_v[a,1:], label=alg_type[i], marker=markers[i], linestyle='none')
    plt.ylabel('user delay')
    plt.xlabel(f'lambda req/s')
    plt.legend()
    plt.show()
    plt.pause(0.1)
    
    plt.figure(2)
    for i in range(a+1):
        line, = plt.plot(lambda_v[a,:], rhoce_v[a,:], label=alg_type[i], marker=markers[i])
    plt.ylabel('metwork load cloud edge')
    plt.xlabel(f'lambda req/s')
    plt.legend()
    plt.show()
    plt.pause(0.1)

    #time.sleep(1e6)


