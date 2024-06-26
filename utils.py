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

def computeCost(Acpu,Amem,Qcpu,Qmem,Cost_cpu,Cost_mem):

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
    cloud_cpu_reduction[np.isnan(cloud_cpu_reduction)] = 0
    cloud_mem_reduction[np.isnan(cloud_mem_reduction)] = 0
    cloud_cpu_reduction[cloud_cpu_reduction==-np.inf] = 0
    cloud_mem_reduction[cloud_mem_reduction==-np.inf] = 0
    Acpu_new[M:] = Acpu_new[M:] + cloud_cpu_reduction # edge cpu increase
    Amem_new[M:] = Amem_new[M:] + cloud_mem_reduction # edge mem increase
    Acpu_new[:M] = Acpu_new[:M] - cloud_cpu_reduction # cloud cpu decrease
    Amem_new[:M] = Amem_new[:M] - cloud_mem_reduction # cloud mem decrease

    return Acpu_new, Amem_new