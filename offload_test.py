# pylint: disable=C0103, C0301
from SAMP_offload import offload
from MFU_heuristic import mfu_heuristic
import argparse
import logging
import sys
import numpy as np
import networkx as nx
import utils
import random
import time
from computeN import computeN
from computeDTot import computeDTot
from buildFi import buildFi
import json



# MAIN
def main():
    # small simulation to test the offload function
    #strategy = offload
    strategy = mfu_heuristic
    RTT = 0.106    # RTT edge-cloud
    M = 200 # n. microservices
    delay_decrease_target = 0.05    # requested delay reduction
    lambda_val = 100     # request per second
    B = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the edge istio proxy must be set equal to one

    Cost_cpu_edge = 0.056/2 # cost of CPU at the edge per hour
    Cost_mem_edge = 0.056/4 # cost of 1 GB memory at the edge per hour
    Cost_cpu_cloud = 0.0416/2 # cost of CPU at the edge per hour
    Cost_mem_cloud = 0.0416/4 # cost of 1 GB memory at the edge per hour
    Cost_network = 0.02 # cost of network traffic per GB

    Fm_range_min = 0.1 # min value of microservice call frequency 
    Fm_range_max = 0.5 # max value of microservice call frequency 
    Ucpu_range_min = 1  # min value of requested CPU quota per instance-set per 10 req/s
    Ucpu_range_max = 128 # max value of requested CPU quota per instance-set per 10 req/s
    L_range_min = 1e6 # min value of response size in bytes
    L_range_max = 1e6   # max of response size in bytes
    
    Di = np.zeros(2*M)
    
    if L_range_min == L_range_max:
        L = np.ones(M) * L_range_min
    else:
        L = np.random.randint(L_range_min,L_range_max,M)  # random response size bytes
    L[M-1]=0 # istio proxy has no response size
    
    # build dependency graph
    random=dict()
    random['n_parents'] = 2
    Fm = np.zeros([M,M])   # microservice call frequency matrix
    for i in range(1,M-1):
        if random['n_parents']>1:
            n_parent=np.random.randint(1,random['n_parents'])
        else:
            n_parent=1
        for j in range(n_parent):
            a = np.random.randint(i)
            Fm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fm[i,j]=np.random.uniform(Fm_range_min,Fm_range_max) if Fm[i,j]>0 else 0
    Fm[M-1,0] = 1  # istio proxy / user call microservice 0 (the ingress microservice)
    
    
    # set random values for CPU and memory requests
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
    

    utils.computeResourceShift(Ucpu_void,Umem_void,N_void,Ucpu_void,Umem_void,N_void)
    delay_old,_,_,rhoce_old = computeDTot(S_b_void, N_void, Fi_void, Di, np.tile(L, 2), RTT, B, lambda_val, M)
    Cost_old,Cost_edge_old,Cost_cloud_old,Cost_traffic_ce_old = utils.computeCost(Ucpu_void, Umem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_old * B, Cost_network)
    # set 0 random internal delay
    # print(f"{Cost_old},{Cost_edge_old},{Cost_cpu_edge_old},{Cost_mem_edge_old},{Cost_cloud_old},{Cost_cpu_cloud_old},{Cost_mem_cloud_old},{Cost_traffic_ce_old}")
    
    # Call the offload function
    params = {
        'S_edge_b': S_edge_b,
        'Ucpu': Ucpu_void,
        'Umem': Umem_void,
        'Fm': Fm,
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
        'locked': None,
        'traces_b': None,
        'Qcpu': Qcpu,
        'Qmem': Qmem,
        'mode': 'offload',
        'expanding-depth':2,
        'max-sgs': 128,
        'max-traces': 2048,
        'sgs-builder': 'sgs_builder_traces'
    }

    tic = time.time()    
    result_list = strategy(params)
    toc = time.time()
    result=result_list[1]
    print(f"Initial config:\n edge microservices: {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_old}")
    print(f"Result for offload:\n edge microservices: {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")
    #print(json.dumps(result, indent=4))
    print(f'processing time {(toc-tic)} sec')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '-log',
                     '--loglevel',
                     default='info',
                     help='Provide logging level. Example --loglevel debug, default=info' )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(asctime)s EPAMP offload %(levelname)s %(message)s')

    logging.info( 'Logging now setup.' )
    # Define the input variables
    seed = 150271
    np.random.seed(seed)
    random.seed(seed)
    main()

