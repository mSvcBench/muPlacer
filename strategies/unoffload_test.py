# pylint: disable=C0103, C0301

import logging
import sys
import argparse
import numpy as np
import networkx as nx
import random

from utils import buildFi, computeDTot, computeN, computeCost, computeResourceShift
from numpy import inf
from SBMP_unoffload import sbmp_u
from MFU import mfu


def main():
    # small simulation to test the unoffload function
    strategy = sbmp_u

    RTT = 0.106    # RTT edge-cloud
    M = 100 # n. microservices
    delay_increase_target = 0.01    # requested delay reduction
    lambda_val = 50     # request per second
    B = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one

    Cost_cpu_edge = 0.056/2 # cost of CPU at the edge per hour
    Cost_mem_edge = 0.056/4 # cost of 1 GB memory at the edge per hour
    Cost_cpu_cloud = 0.0416/2 # cost of CPU at the edge per hour
    Cost_mem_cloud = 0.0416/4 # cost of 1 GB memory at the edge per hour
    Cost_network = 0.02 # cost of network traffic per GB

    random=dict()
    random['n_parents'] = 2

    Fm_range_min = 0.1 # min value of microservice call frequency 
    Fm_range_max = 0.5 # max value of microservice call frequency 
    Ucpu_range_min = 1  # min value of requested CPU quota per instance-set
    Ucpu_range_max = 32 # max value of requested CPU quota per instance-set
    L_range_min = 1000 # min value of response size in bytes
    L_range_max = 50000   # max of response size in bytes
    
    L = np.random.randint(L_range_min,L_range_max,M)  # random response size bytes
    L[M-1]=0 # istio ingress has no response size
    
    # build dependency graph
    Fm = np.zeros([M,M])   # microservice call frequency matrix
    for i in range(1,M-1):
        n_parent=np.random.randint(1,random['n_parents'])
        for j in range(n_parent):
            a = np.random.randint(i)
            Fm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fm[i,j]=np.random.uniform(Fm_range_min,Fm_range_max) if Fm[i,j]>0 else 0
    Fm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)
    
    # create a cloud-only deployment (void)
    Ucpu_void = np.random.uniform(Ucpu_range_min,Ucpu_range_max,size=M) * lambda_val / 10
    Ucpu_void[M-1]=0   # istio proxy has no CPU request
    Ucpu_void = np.concatenate((Ucpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Umem_void = Ucpu_void * 2 # memory request is twice the CPU request (rule of tumb 1 CPU x 2GBs)
    Qcpu = np.ones(2*M)   # CPU quantum in cpu sec
    Qmem = np.zeros(2*M)   # Memory quantum in bytes
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fi_void = np.matrix(buildFi(S_b_void, Fm, M))    # instance-set call frequency matrix of the void state
    N_void = computeN(Fi_void, M, 2)    # number of instance call per user request of the void state
    
    # create a random edge conf by adding x dependency paths at random
    x = 10
    G = nx.DiGraph(Fm) # Create microservice dependency graph 
    dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
    for ms in range(M-1):
        paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        for path_n in paths_n:
            path_b = np.zeros((1,M),int)
            path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
            dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    l = len(dependency_paths_b)
    random_values = np.random.choice(range(l), size=x, replace=False)
    for j in random_values:
        S_edge_b = np.minimum(S_edge_b + dependency_paths_b[j],1)
    S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
    S_b[M-1] = 0  # edge istio proxy (user)
    
    # compute Ucpu and Umem for the current state
    # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
    Fi = np.matrix(buildFi(S_b, Fm, M))    # instance-set call frequency matrix of the current state
    N = computeN(Fi, M, 2)    # number of instance call per user request of the current state
    Ucpu = Ucpu_void.copy()
    Umem = Umem_void.copy()
    Di = np.zeros(2*M)
    delay,_,_,rhoce = computeDTot(S_b, N, Fi, Di, np.tile(L, 2), RTT, B, lambda_val, M, np.empty(0))
    computeResourceShift(Ucpu,Umem,N,Ucpu_void,Umem_void,N_void)           
    Cost_edge = computeCost(Ucpu, Umem, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce * B, Cost_network)[0]
    
    # Call the unoffload function
    params = {
        'S_edge_b': S_edge_b,
        'Ucpu': Ucpu,
        'Umem': Umem,
        'Qcpu': Qcpu,
        'Qmem': Qmem, 
        'Fm': Fm,
        'M': M,
        'lambd': lambda_val,
        'L': L,
        'Di': Di,
        'delay_increase_target': delay_increase_target,
        'RTT': RTT,
        'B': B,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'Cost_cpu_cloud': Cost_cpu_cloud,
        'Cost_mem_cloud': Cost_mem_cloud,
        'Cost_network': Cost_network,
        'locked_b': None,
        'expanding-depth':2,
        'max-sgs': 128,
        'max-traces': 2048,
        'sgs-builder': 'sgs_builder_traces',
        'mode': 'unoffload'
    }
    
    # Call the unoffload function
    result_list = strategy(params)
    result=result_list[2]
    print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for unoffload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '-log',
                     '--loglevel',
                     default='warning',
                     help='Provide logging level. Example --loglevel debug, default=warning' )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(asctime)s SBMP offload %(levelname)s %(message)s')
    logging.info( 'Logging now setup.' )

    seed = 150271
    np.random.seed(seed)
    random.seed(seed)

    main()
