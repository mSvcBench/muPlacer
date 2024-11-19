import numpy as np

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
    # HPA_cpu_th : CPU threshold for HPA
    M = int(len(Ucpu)/2)
    if HPA_cpu_th is None:
        HPA_cpu_th = np.ones(M) * 0.6 # default value
    scaled_Ucpu = np.divide(Ucpu,np.tile(HPA_cpu_th,2))
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
        HPA_cpu_th = np.ones(M) * 0.6 # default value
    
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

# def computeCost_old(Acpu,Amem,Qcpu,Qmem,Cost_cpu,Cost_mem):

#     # Compute the cost of a configuration 
#     # Acpu : actual CPU request of microservices
#     # Amem : actual Memory request of microservices
#     # Qcpu : CPU quota of microservices
#     # Qmem : Memory quota of microservices
#     # Cost_cpu : CPU cost 
#     # Cost_mem : Memory cost
    
#     Acpu_sum = np.sum(qz(Acpu, Qcpu))
#     Amem_sum = np.sum(qz(Amem, Qmem))
#     Cost_cpu_sum = Cost_cpu * Acpu_sum
#     Cost_mem_sum = Cost_mem * Amem_sum
#     Cost_sum = Cost_cpu_sum + Cost_mem_sum
#     return Cost_sum,Cost_cpu_sum,Cost_mem_sum