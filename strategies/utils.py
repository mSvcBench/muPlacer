import numpy as np
from math import log2
import logging

def numpy_array_to_list(numpy_array):
    return list(numpy_array.flatten())

# def qz(x,y):
#     # return the ceil of x/y if y>0, x otherwise
#     # used to compute the  resource rquest of microservices considering the quota of instances. In the paper we compute replicas and cost per replicas. With qz we obtain the same value
#     if isinstance(x, np.ndarray):
#         res = np.zeros(len(x))
#         z = np.argwhere(y==0)
#         res[z] = x[z]
#         nz = np.argwhere(y>0)
#         res[nz] = np.ceil(x[nz]/y[nz])*y[nz]
#         return res
#     else:
#         if y == 0:
#             return x
#         else:
#             return np.ceil(x/y)*y


def computeReplicas(Ucpu,Qcpu,Qmem,HPA_cpu_th=None):
    # Compute the number of replicas of a configuration 
    # Ucpu : CPU usage of microservices
    # Qcpu : CPU quota (request) of microservice istances
    # Qmem : Memory quota (request) of microservice istances
    # HPA_cpu_th : CPU utilization threshold for HPA
    M = int(len(Ucpu)/2)
    if HPA_cpu_th is None:
        HPA_cpu_th = np.ones(2*M) * 0.6 # default value 0.6
    scaled_Ucpu = np.divide(Ucpu,HPA_cpu_th)
    Rp_cloud = np.ceil(scaled_Ucpu[:M]/Qcpu[:M])
    Rp_cloud = np.maximum(Rp_cloud,1)
    Rp_edge = np.ceil(scaled_Ucpu[M:]/Qcpu[M:])
    # replace nan value with zero
    Rp_edge = np.nan_to_num(Rp_edge)
    Rp_cloud = np.nan_to_num(Rp_cloud) 

    return Rp_cloud, Rp_edge
      
def computeCost(Ucpu,Umem,Qcpu,Qmem,Cost_cpu_edge,Cost_mem_edge,Cost_cpu_cloud,Cost_mem_cloud,Tnce,Cost_network,HPA_cpu_th=None):

    # Compute the cost of a configuration 
    # Ucpu : CPU usage of microservices
    # Umem : Memory usage of microservices
    # Qcpu : CPU quota (request) of microservice istances
    # Qmem : Memory quota (request) of microservice istances
    # Cost_cpu_edge : CPU cost edge per hour
    # Cost_mem_edge : Memory cost edge per hour
    # Cost_cpu_cloud : CPU cost cloud per hour
    # Cost_mem_cloud : Memory cost cloud per hour
    # Tnce : network traffic cloud-edge bit/s
    # Cost_network : network cost per GB
    # HPA_cpu_th : CPU threshold for HPA
    M = int(len(Ucpu)/2)
    if HPA_cpu_th is None:
        HPA_cpu_th = np.ones(2*M) * 0.6 # default value
    
    # Compute the cost per instance
    Ci_cloud = Cost_cpu_cloud * Qcpu[:M] + Cost_mem_cloud * Qmem[:M]
    Ci_edge = Cost_cpu_edge * Qcpu[M:] + Cost_mem_edge * Qmem[M:]

    # compute replicas
    Rp_cloud, Rp_edge = computeReplicas(Ucpu,Qcpu,Qmem,HPA_cpu_th)

    # Compute the cost of the configuration
    Cost_edge = np.sum(np.multiply(Rp_edge, Ci_edge))
    Cost_cloud = np.sum(np.multiply(Rp_cloud, Ci_cloud))
    Cost_traffic_ce = Cost_network * Tnce * 3600/8/1e9 # traffic cost per hour
    Cost_sum = Cost_edge + Cost_cloud + Cost_traffic_ce    # Total cost per hour

    return Cost_sum, Cost_edge, Cost_cloud, Cost_traffic_ce

def computeResourceShift(Ucpu_new,Umem_new,N_new,Ucpu_old,Umem_old,N_old):
    # Compute the resource shift of a configuration
    # Ucpu_old : old CPU usage of microservices
    # Umem_old : old Memory usage of microservices
    # N_old : old number of instance call per user request
    # N_new : new number of instance call per user request
    # Ucpu_new : new CPU usage of microservices
    # Umem_new : new Memory usage of microservices

    M = int(len(N_new)/2)
    np.copyto(Ucpu_new,Ucpu_old)
    np.copyto(Umem_new,Umem_old)

    Ucpu_co = Ucpu_old[:M]+Ucpu_old[M:]   # cpu usage of a cloud only deployment
    Umem_co = Umem_old[:M]+Umem_old[M:]   # memory usage of a cloud only deployment
    Nt = N_old[:M]+N_old[M:]  # total number of microservice call per user request
    Ucpu_new[:M] = np.multiply(Ucpu_co, np.divide(N_new[:M] , Nt))  # cloud cpu usage
    Ucpu_new[M:] = np.multiply(Ucpu_co, np.divide(N_new[M:] , Nt))  # edge cpu usage
    Umem_new[:M] = np.multiply(Umem_co, np.divide(N_new[:M] , Nt))  # cloud memory usage
    Umem_new[M:] = np.multiply(Umem_co, np.divide(N_new[M:] , Nt))  # edge memory usage
    
    return Ucpu_new, Umem_new

def sgs_builder_traces_full(M,max_traces,Fm):
    # Create a random trace
    # M : number of microservices
    # max_traces : maximum number of traces
    # Fm : microservice call frequency matrix
    n_traces = max_traces
    expanding_subgraphs_b_full = np.empty((0,M), int)
    user = M-1
    iteration = 0
    while True:
        iteration += 1
        trace_sample_b = np.zeros(M)
        trace_sample_b = sgs_builder_trace(user,trace_sample_b,Fm)
        expanding_subgraphs_b_full = np.append(expanding_subgraphs_b_full, trace_sample_b.reshape(1, -1), axis=0)
        if len(expanding_subgraphs_b_full) >= n_traces or (iteration > 100*n_traces and len(expanding_subgraphs_b_full) > 20):
            break
    trace_sample_b = np.ones(M)  # add full edge trace
    expanding_subgraphs_b_full = np.append(expanding_subgraphs_b_full, trace_sample_b.reshape(1, -1), axis=0)
    return expanding_subgraphs_b_full

def sgs_builder_trace(node,trace,Fm):
    children = np.argwhere(Fm[node,:]>0).flatten()
    for child in children:
        if np.random.random() < Fm[node,child]:
            trace[child] = 1
            trace = sgs_builder_trace(child,trace,Fm)
    return trace

def buildFi(S, Fm, M):
    #   Create instance-level call frequency matrix Fi  

    #   S : vector id Sid, S[:M] (S[M:]) binary encoded presence vector for cloud (edge) microservices
    #   M : number of microservices
    #   Fm : microservice-level call frequency matrix
    #   Fi : instance-level call frequency matrix 
    MN = M * 2   # mesh nodes
    Fi = np.zeros((MN, MN))
    Fi[:M-1,:M-1] = Fm[:M-1,:M-1]
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    # temporary initialize all edge instances call cloud instances
    Fi[M:,:M] = Fm[:,:]
    Fi[M+S_not_edge_id,:] = 0 # clean rows of edge instance not present @ edge
    y = np.repeat([S_edge_id], len(S_edge_id),axis=0).T
    Fi[M+S_edge_id,M+y] = Fm[S_edge_id,y] # edge instances call edge insances
    Fi[M+S_edge_id,y] = 0 # clean edge cloud call for instances both at the edge 
    return Fi

def computeDiTot(N, Di):

    # compute total internal delay introduced by microserive per user request
    # N : number of instance call per user request of the current state
    # Di : internal delay of microservices
    
    return (np.sum(np.multiply(N,Di)))

def computeDnTot(S, N, Fi, L, RTT, B, lambd, M, Ld = np.empty(0)):

    # compute cloud-edge traffic
    
    # S : binary presence vector
    # N : average number of calls per user request per microservice
    # Fi : call frequency matrix
    # L : response length of microservices
    # RTT : round trip time
    # B : network bandwidth
    # lambd : average number of requests per second
    # M : number of microservices
    # Ld : duration of cloud edge data transfer for L 

    max_delay = 1e5 # max delay used to avoid inf problem during optimization
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    if Ld.size==0:
        Tnce = np.sum(np.multiply(np.multiply(Fi[M:,:M],np.repeat(N[M:].reshape(M,1),M,axis=1)),np.repeat(L[:M].reshape(1,M),M,axis=0)))*lambd*8
        rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection
        load_spread = 0
        rhonce_max = 1
        if rhonce < rhonce_max:
            load_spread = 1/(1 - rhonce)  # Load spread factor
        else:
            load_spread = 1e6*Tnce/B    # Load spread factor fostering solution with lower traffic
            # load_spread = (rhonce * 1/((1-rhonce_max)**2)) + ((1-2*rhonce_max)/((1-rhonce_max)**2))  # Load spread factor
        Dn = np.repeat(np.minimum(((L * 8 / B)*(load_spread)),max_delay).reshape(1,2*M),2*M,axis=0)+RTT
    else:
        Dn[M:,:M]=np.repeat(Ld.reshape(1,M),M,axis=0)
    Dn = np.repeat(np.minimum(((L * 8 / B)*(load_spread)),max_delay).reshape(1,2*M),2*M,axis=0)+RTT
    Dn_tot = np.sum(np.multiply((N[M:].reshape(M,1)),(np.sum(np.multiply(Fi[M:,:M],Dn[M:,:M]),axis=1))))

    return Dn_tot, rhonce 

def computeDTot(S, N, Fi, Di, L, RTT, B, lambd, M, Ld = np.empty(0)):

    # compute average service delay measured at the ingress proxy of the edge data center
    
    # S : binary presence vector
    # N : average number of calls per user request per microservice
    # Fi : call frequency matrix
    # Di : internal delay introduced by microservices
    # L : response length of microservices
    # RTT : round trip time
    # B : network bandwidth
    # lambd : average number of requests per second
    # M : number of microservices
    # Ld : duration of cloud edge data transfer for L
    
    max_delay = 1e5 # max delay used to avoid inf problem during optimization
    Dn_tot, rhonce = computeDnTot(S, N, Fi, L, RTT, B, lambd, M, Ld)
    Di_tot = computeDiTot(N, Di)
    if rhonce == 1:
        logging.debug(f"computeDTot: inf network delay")
    return min(Dn_tot + Di_tot,max_delay), Di_tot, Dn_tot, rhonce

def computeN(Fc, M, e):
    
    #   Compute average number of time a microservice/instance is called per request
    #   M : number of microservices
    #   e : binary indicator: e=1 compute Nc per microservice, e=2 compute Nc per instance
    #   Fc : call frequency matrix 

    MN = len(Fc)
    H = -Fc.T.copy()
    np.fill_diagonal(H, 1)
    if e > 1:
        Ubit = np.arange(2, e+1) * M  # user position in the state vector
    else:
        Ubit = MN
    N = np.zeros(MN)
    N[Ubit-1] = 1

    Nc = np.linalg.solve(H,N)
    Nc = np.array(Nc).flatten()
    return Nc

def id2S(Sid, Ns):
    S = list(bin(Sid - 1)[2:])
    S = list(map(int, S))
    S = [0] * (int(log2(Ns)) - len(S)) + S
    return S

def mV2mI(i, di, M):
    # Convert the vector representation <i,di> of a microservice to its id I
    # i: number of the microservice
    # di: datacenter of the microservice (1,2,3...)
    # M: number of microservices
    
    I = i + (di-1)*M  # id of the datacenter. (1, cloud, 2 edge1, 3 edge2,...)
    return I

def mI2mV(I, M):
    # Convert the id of a service I to its vector representation <i,di>
    # I: id of the microservice
    # M: number of microservices
    di = (I // M) + 1  # id of the datacenter. (1, cloud, 2 edge1, 3 edge2,...)
    i = I - (di - 1) * M  # id of the microservice
    return i, di

def S2id(S):
    # convert state value to decimal state id
    S = [int(x) for x in S]
    num_bin = ''.join(map(str, S))
    Sid = int(num_bin, 2) + 1
    return Sid

def netdelay(S, RTT, B, lambd, L, Fi, N, M, e):
    MN = M * e  # edge+cloud microservice instance-sets
    Tnce = np.zeros(e - 1)  # Inizialization of array for volume of cloud-edge traffic
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e + 1):  
                if di == h and dj == 1 and S[I]==1:
                    Tnce[h - 2] = Tnce[h - 2] + lambd * N[I] * Fi[I, J] * L[J] * 8 # Compute Tnce
    
    rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e+1):
                if di == h and dj == 1 and Fi[I, J]>0:
                    Dn[I,J] = RTT + ((L[J] * 8 / B) + 0.015) / (1 - rhonce[h - 2]) 
                    continue
    return Dn, Tnce

def netdelay2(S, RTT, B, lambd, L, Fi, N, M):
    MN = 2*M  # edge+cloud microservice instance-sets
    Tnce = 0  # Inizialization of array for volume of cloud-edge traffic
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    for i in S_edge_id:
        for j in S_not_edge_id:
            Tnce = Tnce + lambd * N[M+i] * Fi[M+i, j] * L[j] * 8 # Compute Tnce
    
    rhonce = min(Tnce / B, 1)  # Utilization factor of the cloud-edge connection

    # Compute Dn
    Dn = np.zeros((MN, MN)) # Inizialization of matrix of network delays
    for i in S_edge_id:
        for j in S_not_edge_id:
            if Fi[M+i,j]>0:
                Dn[M+i,j] = RTT + ((L[j] * 8 / B) + 0.015) / (1 - rhonce) 
    return Dn, Tnce