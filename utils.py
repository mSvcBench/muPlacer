import numpy as np

def numpy_array_to_list(numpy_array):
    return list(numpy_array.flatten())

def qz(x,y):
    if isinstance(x, np.ndarray):
        res = np.zeros(len(x))
        z = np.argwhere(y==0)
        res[z] = x[z]
        nz = np.argwhere(y>0)
        res[nz] = np.ceil(x[nz]/y[nz])*y[nz]
        return res
    else:
        if y == 0:
            return x
        else:
            return np.ceil(x/y)*y

def computeCost_old(Acpu,Amem,Qcpu,Qmem,Cost_cpu,Cost_mem):

    # Compute the cost of a configuration 
    # Acpu : actual CPU request of microservices
    # Amem : actual Memory request of microservices
    # Qcpu : CPU quota of microservices
    # Qmem : Memory quota of microservices
    # Cost_cpu : CPU cost 
    # Cost_mem : Memory cost
    
    Acpu_sum = np.sum(qz(Acpu, Qcpu))
    Amem_sum = np.sum(qz(Amem, Qmem))
    Cost_cpu_sum = Cost_cpu * Acpu_sum
    Cost_mem_sum = Cost_mem * Amem_sum
    Cost_sum = Cost_cpu_sum + Cost_mem_sum
    return Cost_sum,Cost_cpu_sum,Cost_mem_sum

def computeCost(Acpu,Amem,Qcpu,Qmem,Cost_cpu_edge,Cost_mem_edge,Cost_cpu_cloud,Cost_mem_cloud,Tnce,Cost_network):

    # Compute the cost of a configuration 
    # Acpu : actual CPU request of microservices
    # Amem : actual Memory request of microservices
    # Qcpu : CPU quota of microservices
    # Qmem : Memory quota of microservices
    # Cost_cpu_edge : CPU cost edge per hour
    # Cost_mem_edge : Memory cost edge per hour
    # Cost_cpu_cloud : CPU cost cloud per hour
    # Cost_mem_cloud : Memory cost cloud per hour
    # Tnce : network traffic cloud-edge bit/s
    # Cost_network : network cost per GB
    M = int(len(Acpu)/2)
    Acpu_sum_edge = np.sum(qz(Acpu[M:2*M-1], Qcpu[M:2*M-1]))
    Amem_sum_edge = np.sum(qz(Amem[M:2*M-1], Qmem[M:2*M-1]))
    Cost_cpu_sum_edge = Cost_cpu_edge * Acpu_sum_edge # edge cpu cost per hour
    Cost_mem_sum_edge = Cost_mem_edge * Amem_sum_edge # edge mem cost per hour
    Cost_sum_edge = Cost_cpu_sum_edge + Cost_mem_sum_edge # Total edge cost per hour
    Acpu_sum_cloud = np.sum(qz(np.maximum(Acpu[:M-1],Qcpu[:M-1]), Qcpu[:M-1])) # maximum because at least a replica run in the cloud
    Amem_sum_cloud = np.sum(qz(np.maximum(Amem[:M-1],Qmem[:M-1]), Qmem[:M-1])) # maximum because at least a replica run in the cloud
    Cost_cpu_sum_cloud = Cost_cpu_cloud * Acpu_sum_cloud # cloud cpu cost per hour
    Cost_mem_sum_cloud = Cost_mem_cloud * Amem_sum_cloud # cloud mem cost per hour
    Cost_sum_cloud = Cost_cpu_sum_cloud + Cost_mem_sum_cloud # Total cloud cost per hour
    Cost_traffic_ce = Cost_network * Tnce * 3600/8/1e9 # traffic cost per hour
    Cost_sum = Cost_sum_edge + Cost_sum_cloud + Cost_traffic_ce    # Total cost per hour

    return Cost_sum, Cost_sum_edge,Cost_cpu_sum_edge,Cost_mem_sum_edge, Cost_sum_cloud,Cost_cpu_sum_cloud,Cost_mem_sum_cloud, Cost_traffic_ce

def computeResourceShift(Acpu_new,Amem_new,Nci_new,Acpu_old,Amem_old,Nci_old):
    # Compute the resource shift of a configuration
    # Acpu : actual CPU request of microservices
    # Amem : actual Memory request of microservices
    # Nci : number of instance call per user request

    M = int(len(Nci_new)/2)
    np.copyto(Acpu_new,Acpu_old)
    np.copyto(Amem_new,Amem_old)
    
    cloud_cpu_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Acpu_old[:M]
    cloud_mem_reduction = (1-Nci_new[:M]/Nci_old[:M]) * Amem_old[:M]
    #cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    #cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    
    edge_cpu_reduction = (1-Nci_old[:M]/Nci_new[:M]) * Acpu_old[M:]
    edge_mem_reduction = (1-Nci_old[:M]/Nci_new[:M]) * Amem_old[M:]
    #edge_cpu_reduction[np.isnan(edge_cpu_reduction)] = 0
    #edge_mem_reduction[np.isnan(edge_mem_reduction)] = 0
    
    no_cloud_cost1 = np.argwhere(cloud_cpu_reduction==-np.inf)
    no_cloud_cost2 = np.argwhere(np.isnan(cloud_cpu_reduction))
    no_cloud_cost = np.concatenate((no_cloud_cost1,no_cloud_cost2))
    cloud_cpu_reduction[no_cloud_cost] = -edge_cpu_reduction[no_cloud_cost]
    cloud_mem_reduction[no_cloud_cost] = -edge_mem_reduction[no_cloud_cost]
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    
    Acpu_new[M:] = np.round(Acpu_new[M:] + cloud_cpu_reduction,3) # edge cpu increase
    Amem_new[M:] = np.round(Amem_new[M:] + cloud_mem_reduction,3) # edge mem increase
    Acpu_new[:M] = np.round(Acpu_new[:M] - cloud_cpu_reduction,3) # cloud cpu decrease
    Amem_new[:M] = np.round(Amem_new[:M] - cloud_mem_reduction,3) # cloud mem decrease

    return Acpu_new, Amem_new