from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from offload import offload
from mfu_heuristic import mfu_heuristic
from IA_heuristic import IA_heuristic
from mubenchFcmBuilder import createFcm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
from computeNc import computeNc
from scipy.io import savemat
from buildFci import buildFci
from numpy import inf
import time

def edges_reversal(graph):
    for edge in graph.get_edgelist():
        graph.delete_edges([(edge[0], edge[1])])
        graph.add_edges([(edge[1], edge[0])])

RTT = 0.04    # RTT edge-cloud
input_file = "simulators/workmodel.json"
Fcm,Rcpu_x_rep,Rs = createFcm(input_file) # Call frequency matrix, CPU requirements x replica, average response size
M = len(Fcm) # n. microservices + user
delay_decrease_target = 0.04    # requested delay reduction
lambda_val = 20     # request per second
Ne = 100e3    # bitrate cloud-edge
# set random  internal delay equal to 0 since assuming equal computing performance
Di = np.zeros(2*M) # internal delay of microservices
Rmem = np.zeros(2*M) # void memory requirements

# current status of the system
replicas=np.zeros(2*M) # current replica configuration
replicas[0:M] = 1 # at least a replica in the cloud
replicas[M:] = 0 # no replica at the edge
replicas[M-1] = 0 # user 
## current config
replicas[M+0]=0

## end current config
S_edge_b = np.where(replicas[M:] > 0, 1, 0) # current placement state
S_edge_b[M-1] = 1 # Last value is the user must be set equal to one
S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
S_b[M-1] = 0  # User is not in the cloud
Cost_cpu_edge = 1 # cost of CPU at the edge
Cost_mem_edge = 1 # cost of memory at the edge


Rcpu_x_rep = np.repeat(Rcpu_x_rep, 2, axis=0)
Rcpu = np.multiply(Rcpu_x_rep, replicas)
Cost_edge = Cost_cpu_edge * np.sum(Rcpu[M:]) + Cost_mem_edge * np.sum(Rmem[M:]) # Total edge cost of the current state

# Call the offload function
params = {
    'S_edge_b': S_edge_b,
    'Rcpu': Rcpu,
    'Rmem': Rmem,
    'Fcm': Fcm,
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
    'u_limit': 2,
    'no_caching': False
}

        
result = offload(params)
print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}, rounds = {result['n_rounds']}")

