def mI2mV(I, M):
    # Convert the id of a service I to its vector representation <i,di>
    # I: id of the microservice
    # M: number of microservices
    di = (I // M) + 1  # id of the datacenter. (1, cloud, 2 edge1, 3 edge2,...)
    i = I - (di - 1) * M  # id of the microservice
    return i, di