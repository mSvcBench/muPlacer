# pylint: disable=C0103, C0301
from EPAMP_offload_sweeping import offload
from MFU_heuristic import mfu_heuristic
import argparse
import logging
import sys
import numpy as np
import networkx as nx
import utils
import random
import time
from computeNc import computeNc
from computeDTot import computeDTot
from buildFci import buildFci
import json



# MAIN
def main():
    # small simulation to test the offload function
    strategy = offload
    RTT = 0.106    # RTT edge-cloud
    M = 200 # n. microservices
    delay_decrease_target = 0.05    # requested delay reduction
    lambda_val = 100     # request per second
    Ne = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the edge istio proxy must be set equal to one

    Cost_cpu_edge = 0.056/2 # cost of CPU at the edge per hour
    Cost_mem_edge = 0.056/4 # cost of 1 GB memory at the edge per hour
    Cost_cpu_cloud = 0.0416/2 # cost of CPU at the edge per hour
    Cost_mem_cloud = 0.0416/4 # cost of 1 GB memory at the edge per hour
    Cost_network = 0.02 # cost of network traffic per GB

    Fcm_range_min = 0.1 # min value of microservice call frequency 
    Fcm_range_max = 0.5 # max value of microservice call frequency 
    Acpu_range_min = 1  # min value of requested CPU quota per instance-set per 10 req/s
    Acpu_range_max = 128 # max value of requested CPU quota per instance-set per 10 req/s
    Rs_range_min = 1e6 # min value of response size in bytes
    Rs_range_max = 1e6   # max of response size in bytes
    
    Di = np.zeros(2*M)
    
    if Rs_range_min == Rs_range_max:
        Rs = np.ones(M) * Rs_range_min
    else:
        Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # istio proxy has no response size
    
    # build dependency graph
    random=dict()
    random['n_parents'] = 2
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    for i in range(1,M-1):
        if random['n_parents']>1:
            n_parent=np.random.randint(1,random['n_parents'])
        else:
            n_parent=1
        for j in range(n_parent):
            a = np.random.randint(i)
            Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(Fcm_range_min,Fcm_range_max) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # istio proxy / user call microservice 0 (the ingress microservice)
    
    
    # set random values for CPU and memory requests
    Acpu_void = np.random.uniform(Acpu_range_min,Acpu_range_max,size=M) * lambda_val / 10
    Acpu_void[M-1]=0   # istio proxy has no CPU request
    Acpu_void = np.concatenate((Acpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Amem_void = Acpu_void * 2 # memory request is twice the CPU request (rule of tumb 1 CPU x 2GBs)
    Qcpu = np.ones(2*M)   # CPU quantum in cpu sec
    Qmem = np.zeros(2*M)   # Memory quantum in bytes
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
    

    utils.computeResourceShift(Acpu_void,Amem_void,Nci_void,Acpu_void,Amem_void,Nci_void)
    delay_old,_,_,rhoce_old = computeDTot(S_b_void, Nci_void, Fci_void, Di, np.tile(Rs, 2), RTT, Ne, lambda_val, M)
    Cost_old,Cost_edge_old,Cost_cpu_edge_old,Cost_mem_edge_old, Cost_cloud_old,Cost_cpu_cloud_old,Cost_mem_cloud_old, Cost_traffic_ce_old = utils.computeCost(Acpu_void, Amem_void, Qcpu, Qmem, Cost_cpu_edge, Cost_mem_edge, Cost_cpu_cloud, Cost_mem_cloud, rhoce_old * Ne, Cost_network)
    # set 0 random internal delay
    # print(f"{Cost_old},{Cost_edge_old},{Cost_cpu_edge_old},{Cost_mem_edge_old},{Cost_cloud_old},{Cost_cpu_cloud_old},{Cost_mem_cloud_old},{Cost_traffic_ce_old}")
    
    # Call the offload function
    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Acpu_void,
        'Amem': Amem_void,
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
        'Cost_cpu_cloud': Cost_cpu_cloud,
        'Cost_mem_cloud': Cost_mem_cloud,
        'Cost_network': Cost_network,
        'locked': None,
        'dependency_paths_b': None,
        'Qcpu': Qcpu,
        'Qmem': Qmem,
        'mode': 'offload',
        'u_limit':2,
        'max_dps': 128,
        'max_traces': 1024,
        'dp_builder': 'dp_builder_traces'
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

