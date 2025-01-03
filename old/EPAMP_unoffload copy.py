# pylint: disable=C0103, C0301

import logging
import sys
import argparse
import numpy as np
import networkx as nx
import strategies.utils as utils
from numpy import inf
from computeNc import computeN
from buildFi import buildFi
from computeDTot import computeDTot
from SAMP_offload import offload


np.seterr(divide='ignore', invalid='ignore')
# Set up logger
logger = logging.getLogger('EPAMP_unoffload')
logger_stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(logger_stream_handler)
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s EPAMP unoffload %(levelname)s %(message)s'))
logger.propagate = False

def unoffload(params):

    ## INITIALIZE VARIABLES ##
    #Acpu_old (2*M,) vector of CPU req by instance-set at the cloud (:M) and at the edge (M:)
    #Amem_old (2*M,) vector of Memory req by instance-set at the cloud (:M) and at the edge (M:)
    #Fcm (M,M)microservice call frequency matrix
    #M number of microservices
    #lambd user request rate
    #Rs (M,) vector of response size of microservices
    #S_edge_old (M,) vector of binary values indicating if the microservice is at the edge or not
    #delay_decrease_target delay reduction target
    #RTT fixed delay to add to microservice interaction in addition to the time depending on the response size
    #Ne cloud-edge network bitrate
    #Cost_cpu_edge cost of CPU at the edge
    #Cost_mem_edge cost of Memory at the edge
    #u_limit maximum number of microservices upgrade to consider in the greedy iteraction (lower reduce optimality but increase computaiton speed)


    # mandatory paramenters
    S_edge_old = params['S_edge_b']
    Acpu_old = params['Acpu']
    Amem_old = params['Amem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    delay_increase_target = params['delay_increase_target']
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']

    
    # optional paramenters
    Di = params['Di'] if 'Di' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    max_added_dp = params['max_added_dp'] if 'max_added_dp' in params else 1000000
    min_added_dp = params['min_added_dp'] if 'min_added_dp' in params else 0
    dependency_paths_b = params['dependency_paths_b'] if 'dependency_paths_b' in params else None
    locked = params['locked'] if 'locked' in params else None
    u_limit = params['u_limit'] if 'u_limit' in params else M
    no_evolutionary = params['no_evolutionary'] if 'no_evolutionary' in params else False

    
    S_cloud_old = np.ones(int(M)) # EPAMP assumes all microservice instance run in the cloud
    S_cloud_old[M-1] = 0  # # M-1 and 2M-1 are associated to the edge ingress gateway, therefore M-1 must be set to 0 and 2M-1 to 1
    S_b_old = np.concatenate((S_cloud_old, S_edge_old)) # (2*M,) Initial status of the instance-set in the edge and cloud. (:M) binary presence at the cloud, (M:) binary presence at the edge

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to support matrix operations
    Fci_old = np.matrix(buildFi(S_b_old, Fcm, M)) # (2*M,2*M) instance-set call frequency matrix
    Nci_old = computeN(Fci_old, M, 2)  # (2*M,) number of instance call per user request
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)[0]  # Total delay of the current configuration. It includes only network delays
    delay_target = delay_old + delay_increase_target
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total edge cost of the current state

    S_edge_void = np.zeros(int(M))  # (M,) edge state with no instance-set in the edge
    S_edge_void[M-1] = 1  # edge istio proxy
    S_cloud_void = np.ones(int(M))
    S_cloud_void[M-1] = 0
    S_b_void = np.concatenate((S_edge_void, S_edge_void)) # (2*M,) state with no instance-set in the edge

    Acpu_void = np.zeros(2*M)
    Amem_void = np.zeros(2*M)
    Acpu_void[:M] = Acpu_old[:M]+Acpu_old[M:]
    Acpu_void[M:] = np.zeros(M)
    Amem_void[:M] = Amem_old[:M]+Amem_old[M:]
    Amem_void[M:] = np.zeros(M)

    Fci_void = np.matrix(buildFi(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeN(Fci_void, M, 2)    # number of instance call per user request of the void state
    delay_void = computeDTot(S_b_void, Nci_void, Fci_void, Di, Rs, RTT, Ne, lambd, M)[0]
    delay_decrease_target = max(delay_void - delay_target,0)

    ## BUILDING OF DEPENDENCY PATHS ##
    if dependency_paths_b is None:
        G = nx.DiGraph(Fcm) # Create microservice dependency graph 
        dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths

        ## COMPUTE DEPENDENCY PATHS WITH ALL MICROSERIVES AT THE EDGE##
        for ms in range(M-1):
            paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms))
            for path_n in paths_n:
                # path_n numerical id (n) of the microservices of the dependency path
                # If not all microservices in the path are in the edge this path is not a edge-only
                if not all(S_b_old[M+np.array([path_n])].squeeze()==1):
                    continue
                else:
                    path_b = np.zeros((1,M),int)
                    path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
                    dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    params = {
        'S_edge_b': S_edge_void.copy(),
        'Acpu': Acpu_void.copy(),
        'Amem': Amem_void.copy(),
        'Qcpu': Qcpu,
        'Qmem': Qmem,
        'Fcm': Fcm.copy(),
        'M': M,
        'lambd': lambd,
        'Rs': Rs,
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': locked,
        'dependency_paths_b': dependency_paths_b,
        'u_limit': u_limit,
        'max_added_dp': max_added_dp,
        'min_added_dp': min_added_dp,
        'no_evolutionary': no_evolutionary,
    }
    logger.info(f"unoffload calls offload with void edge and delay_decrease_target: {delay_decrease_target}")
    result_list = offload(params)
    result=result_list[1]
    result['delay_increase'] = (delay_void-result['delay_decrease']) - delay_old
    result['cost_decrease'] = Cost_edge_old-result['Cost']
    result['to-apply'] = utils.numpy_array_to_list(np.argwhere(result['S_edge_b']-S_b_old[M:]>0))
    result['to-delete']= utils.numpy_array_to_list(np.argwhere(S_b_old[M:]-result['S_edge_b']>0))
    del result['delay_decrease']
    del result['cost_increase']
    if result['delay_increase'] < delay_increase_target:
        logger.warning(f"unoffload: delay increase target not reached")
    message = f"Result for unoffload - edge microservice ids: {result['placement']}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}"
    result['info'] = message
    return result_list

def main():
    # small simulation to test the unoffload function

    # Define the input variables
    np.random.seed(150273)
    RTT = 0.0869    # RTT edge-cloud
    M = 30 # n. microservices
    delay_increase_target = 0.03    # requested delay reduction
    lambda_val = 20     # request per second
    Ne = 1e9    # bitrate cloud-edge
    
    S_edge_b = np.zeros(M)  # initial state. 
    S_edge_b[M-1] = 1 # Last value is the user must be set equal to one

    Cost_cpu_edge = 1 # cost of CPU at the edge
    Cost_mem_edge = 1 # cost of memory at the edge

    random=dict()
    random['n_parents'] = 3

    Fcm_range_min = 0.1 # min value of microservice call frequency 
    Fcm_range_max = 0.5 # max value of microservice call frequency 
    Acpu_quota = 0.5    # CPU quota
    Acpu_range_min = 1  # min value of requested CPU quota per instance-set
    Acpu_range_max = 32 # max value of requested CPU quota per instance-set
    Rs_range_min = 1000 # min value of response size in bytes
    Rs_range_max = 50000   # max of response size in bytes
    
    Rs = np.random.randint(Rs_range_min,Rs_range_max,M)  # random response size bytes
    Rs[M-1]=0 # istio ingress has no response size
    
    # build dependency graph
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    for i in range(1,M-1):
        n_parent=np.random.randint(1,random['n_parents'])
        for j in range(n_parent):
            a = np.random.randint(i)
            Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(0.1,0.5) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)
    
    # add x dependency path at random
    G = nx.DiGraph(Fcm) # Create microservice dependency graph 
    dependency_paths_b = np.empty((0,M), int) # Storage of binary-based (b) encoded dependency paths
    for ms in range(M-1):
        paths_n = list(nx.all_simple_paths(G, source=M-1, target=ms)) 
        for path_n in paths_n:
            path_b = np.zeros((1,M),int)
            path_b[0,path_n] = 1 # Binary-based (b) encoding of the dependency path
            dependency_paths_b = np.append(dependency_paths_b,path_b,axis=0)
    l = len(dependency_paths_b)
    x = 10
    random_values = np.random.choice(range(l), size=x, replace=False)
    for j in random_values:
        S_edge_b = np.minimum(S_edge_b + dependency_paths_b[j],1)
    S_b = np.concatenate((np.ones(M), S_edge_b)) # (2*M,) full state
    S_b[M-1] = 0  # edge istio proxy
    # set random values for CPU and memory requests
    Acpu_void = (np.random.randint(32,size=M)+1) * Acpu_quota
    Acpu_void[M-1]=0   # user has no CPU request
    Acpu_void = np.concatenate((Acpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Amem_void = np.zeros(2*M)
    Qcpu = np.ones(2*M)   # CPU quantum in cpu sec
    Qmem = np.zeros(2*M)   # Memory quantum in bytes
    S_b_void = np.concatenate((np.ones(M), np.zeros(M))) # (2*M,) state with no instance-set in the edge
    S_b_void[M-1] = 0  # User is not in the cloud
    S_b_void[2*M-1] = 1  # User is in the cloud
    Fci_void = np.matrix(buildFi(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
    Nci_void = computeN(Fci_void, M, 2)    # number of instance call per user request of the void state
    
    # compute Acpu and Amem for the current state
    # assumption is that cloud resource are reduced proportionally with respect to the reduction of the number of times instances are called
    Fci = np.matrix(buildFi(S_b, Fcm, M))    # instance-set call frequency matrix of the current state
    Nci = computeN(Fci, M, 2)    # number of instance call per user request of the current state
    Acpu = Acpu_void.copy()
    Amem = Amem_void.copy()
    utils.computeResourceShift(Acpu,Amem,Nci,Acpu_void,Amem_void,Nci_void)
    Cost_edge = utils.computeCost(Acpu[M:], Amem[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]

    # set 0 random internal delay
    Di = np.zeros(2*M)
    
    # Call the unoffload function
    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Acpu,
        'Amem': Amem,
        'Qcpu': Qcpu,
        'Qmem': Qmem, 
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'delay_increase_target': delay_increase_target,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 2,
        'max_added_dp': 1000000,
        'min_added_dp': -1,
    }
    
    # Call the unoffload function
    result_list = unoffload(params)
    result=result_list[1]
    print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}, rounds = {result['n_rounds']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '-log',
                     '--loglevel',
                     default='warning',
                     help='Provide logging level. Example --loglevel debug, default=warning' )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(asctime)s EPAMP offload %(levelname)s %(message)s')
    logging.info( 'Logging now setup.' )
    main()
