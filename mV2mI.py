def mV2mI(i, di, M):
    # Convert the vector representation <i,di> of a microservice to its id I
    # i: number of the microservice
    # di: datacenter of the microservice (1,2,3...)
    # M: number of microservices
    
    I = i + (di-1)*M  # id of the datacenter. (1, cloud, 2 edge1, 3 edge2,...)
    return I



