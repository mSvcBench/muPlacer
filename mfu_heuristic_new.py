from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import utils
from numpy import inf
from computeNc import computeNc
from buildFci import buildFci
from computeDTot import computeDTot
import networkx as nx


def mfu_heuristic(params):
    ## VARIABLES INITIALIZATION ##
        
    S_edge_old = params['S_edge_b']
    Acpu_old = params['Acpu']
    Amem_old = params['Amem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target'] if params['mode'] == 'offload' else 0
    delay_increase_target = params['delay_increase_target'] if params['mode'] == 'unoffload' else 0
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)
    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    locked_b = params['locked_b'] if 'locked_b' in params else np.zeros(M)

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:] ,Cost_cpu_edge, Cost_mem_edge)[0] # Total cost of old state


    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))
    Nci_old = computeNc(Fci_old, M, 2)
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)[0]
    Nc = computeNc(Fcm, M, 1) 

    delay_decrease_new = 0
    delay_increase_new = 0
    S_b_new = S_b_old.copy()
    
    ## OFFLOAD ##
    if params['mode'] == 'offload':
        while delay_decrease_target > delay_decrease_new:
            Nc_max=-1
            argmax = -1
            for i in range(M-1):
                 if Nc[i]>Nc_max and S_b_new[i+M]==0 and locked_b[i]==0:
                    argmax = i
                    Nc_max = Nc[i]
            S_b_new[argmax+M] = 1
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)[0] 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                break
        
    ## UNOFFLOAD  ##
    else:
        delay_target = delay_old + delay_increase_target
        
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

        Fci_void = np.matrix(buildFci(S_b_void, Fcm, M))    # instance-set call frequency matrix of the void state
        Nci_void = computeNc(Fci_void, M, 2)    # number of instance call per user request of the void state
        delay_void = computeDTot(S_b_void, Nci_void, Fci_void, Di, Rs, RTT, Ne, lambd, M)[0]
        delay_decrease_target = max(delay_void - delay_target,0)
        locked_b = np.zeros(M)  # locked microservices binary encoding. 1 if the microservice is locked, 0 otherwise
        locked_b[np.argwhere(S_edge_old==0)] = 1 # microservices that originally where not in the edge are locked
        params = {
            'S_edge_b': S_edge_void.copy(),
            'Acpu': Acpu_void.copy(),
            'Amem': Amem_void.copy(),
            'Qcpu': Qcpu,
            'Qmem': Qmem,
            'Fcm': Fcm.copy(),
            'M': M,
            'lambd': lambd,
            'Rs': Rs[:M],
            'Di': Di,
            'delay_decrease_target': delay_decrease_target,
            'RTT': RTT,
            'Ne': Ne,
            'Cost_cpu_edge': Cost_cpu_edge,
            'Cost_mem_edge': Cost_mem_edge,
            'locked_b': locked_b,
            'mode': 'offload'
        }
        result = mfu_heuristic(params)
        S_b_new = np.ones(2*M)
        S_b_new[M:] = result['S_edge_b']
        

    Acpu_new = np.zeros(2*M)
    Amem_new = np.zeros(2*M)
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)            
    if params['mode'] == 'offload':
        # compute final values
        delay_decrease_new = delay_old - delay_new
        np.copyto(Acpu_new,Acpu_old) 
        np.copyto(Amem_new,Amem_old)
        utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old) 
        Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
        cost_increase_new = Cost_edge_new - Cost_edge_old 

        result = dict()
        result['S_edge_b'] = S_b_new[M:].astype(int)
        result['Cost'] = Cost_edge_new
        result['delay_decrease'] = delay_decrease_new
        result['cost_increase'] = cost_increase_new
        result['Acpu'] = Acpu_new
        result['Amem'] = Amem_new
        result['Fci'] = Fci_new
        result['Nci'] = Nci_new
        result['delay'] = delay_new
        result['di'] = di_new
        result['dn'] = dn_new
        result['rhoce'] = rhoce_new
    else:
        # compute final values
        delay_increase_new = delay_new - delay_old
        np.copyto(Acpu_new,Acpu_old) 
        np.copyto(Amem_new,Amem_old) 
        utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old) 
        Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0]
        cost_decrease_new = Cost_edge_old - Cost_edge_new

        result = dict()
        result['S_edge_b'] = S_b_new[M:].astype(int)
        result['Cost'] = Cost_edge_new
        result['delay_increase'] = delay_increase_new
        result['cost_decrease'] = cost_decrease_new
        result['Acpu'] = Acpu_new
        result['Amem'] = Amem_new
        result['Fci'] = Fci_new
        result['Nci'] = Nci_new
        result['delay'] = delay_new
        result['di'] = di_new
        result['dn'] = dn_new
        result['rhoce'] = rhoce_new

    return result


        
if __name__ == "__main__":
    # Define the input variables
    np.random.seed(150273)
    RTT = 0.0869    # RTT edge-cloud
    M = 30 # n. microservices
    delay_decrease_target = 0.03    # requested delay reduction
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
    Rs[M-1]=0 # user has no response size
    
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
    S_b[M-1] = 0  # User is not in the cloud
    # set random values for CPU and memory requests
    Acpu_void = (np.random.randint(32,size=M)+1) * Acpu_quota
    Acpu_void[M-1]=0   # user has no CPU request
    Acpu_void = np.concatenate((Acpu_void, np.zeros(M))) # (2*M,) vector of CPU requests for void state
    Amem_void = np.zeros(2*M)
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
    cloud_cpu_decrease = (1-Nci[:M]/Nci_void[:M]) * Acpu_void[:M]
    cloud_mem_decrease = (1-Nci[:M]/Nci_void[:M]) * Amem_void[:M]
    cloud_cpu_decrease[np.isnan(cloud_cpu_decrease)] = 0
    cloud_mem_decrease[np.isnan(cloud_mem_decrease)] = 0
    cloud_cpu_decrease[cloud_cpu_decrease==-inf] = 0
    cloud_mem_decrease[cloud_mem_decrease==-inf] = 0
    Acpu[M:] = Acpu[M:] + cloud_cpu_decrease # edge cpu increase
    Amem[M:] = Amem[M:] + cloud_mem_decrease # edge mem increase
    Acpu[:M] = Acpu[:M] - cloud_cpu_decrease # cloud cpu decrease
    Amem[:M] = Amem[:M] - cloud_mem_decrease # cloud mem decrease
    Cost_edge = Cost_cpu_edge * np.sum(Acpu[M:]) + Cost_mem_edge * np.sum(Amem[M:]) # Total edge cost of the current state

    # set 0 random internal delay
    Di = np.zeros(2*M)
    
    # Call the unoffload function
    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Acpu,
        'Amem': Amem,
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
        'mode': 'offload'
    }


    result = mfu_heuristic(params)
    print(f"Initial config:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay decrease: {result['delay_decrease']}, cost increase: {result['cost_increase']}")

    # Call the unoffload function
    params = {
        'S_edge_b': result['S_edge_b'],
        'Acpu': result['Acpu'],
        'Amem': result['Amem'],
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'delay_increase_target': 0.03,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'mode': 'unoffload'
    }

    result = mfu_heuristic(params)
    print(f"Result for unoffload:\n {np.argwhere(S_edge_b==1).squeeze()}, Cost: {Cost_edge}")
    print(f"Result for offload:\n {np.argwhere(result['S_edge_b']==1).squeeze()}, Cost: {result['Cost']}, delay increase: {result['delay_increase']}, cost decrease: {result['cost_decrease']}")