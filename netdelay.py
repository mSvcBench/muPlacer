import numpy as np
from mI2mV import mI2mV

#   Comupute the network delay matrix Dn

#   S : state vector
#   RTT : RTT cloud edge
#   Ne : edge-cloud bit rate
#   lambd : user request frequency
#   Rs : byte lenght of the response of microservices
#   Fci : service mesh call frequency matrix 
#   Nc : number of time a microservice is called per request
#   M : number of microservices
#   e : number of datacenters

def netdelay(S, RTT, Ne, lambd, Rs, Pci, Nc, M, e):
    MN = M * e  # mesh nodes of state vector S
    Tnce = np.zeros(e - 1)  # cloud-edge traffic
    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e + 1):  
                if di == h and dj == 1:
                    Tnce[h - 2] = Tnce[h - 2] + lambd * Nc[I] * Pci[I, J] * Rs[J] * 8 # 8 bits in a byte
    
    rhonce = min(Tnce / Ne, 1)  # utilization of the cloud-edge(h-1) link

    Dn = np.zeros((MN, MN))

    for I in range(MN):
        i, di = mI2mV(I, M)
        for J in range(MN):
            j, dj = mI2mV(J, M)
            for h in range(2, e+1):
                if di == h and dj == 1:
                    Dn[I,J] = RTT + ((Rs[J] * 8 / Ne) + 0.015) / (1 - rhonce[h - 2])
                    continue
    return Dn, Tnce

    # MN = M * e  # mesh nodes (1 microservice x data center)
    # Ns = 2 ** MN  # number of possible states of the state vector S
    # Tnce = np.zeros(e - 1)  # cloud edge traffic for each edge
    # Tnec = np.zeros(e - 1)  # edge cloud traffic for each edge
    # for I in range(MN):
    #     i, di = mI2mV(I, M)
    #     for J in range(MN):
    #         j, dj = mI2mV(J, M)
    #         for h in range(2, e + 1):  
    #             if di == h and dj == 1:
    #                 Tnce[h - 2] = Tnce[h - 2] + lambd * Nc[I] * Pci[I, J] * Rs[J] * 8 # 8 bits in a byte
    #             elif di == 1 and dj == h:
    #                 Tnec[h - 2] = Tnec[h - 2] + lambd * Nc[I] * Pci[I, J] * Rs[J] * 8 # 8 bits in a byte
    # rhonce = min(Tnce / Ne, 1)  # utilization of the cloud-edge(h-1) link
    # rhonec = min(Tnec / Ne, 1)  # utilization of the edge(h-1)-cloud link

    # Dn = np.zeros((MN, MN))

    # for I in range(MN):
    #     i, di = mI2mV(I, M)
    #     for J in range(MN):
    #         j, dj = mI2mV(J, M)
    #         for h in range(2, e+1):
    #             if di == h and dj == 1:
    #                 Dn[I,J] = RTT + ((Rs[J] * 8 / Ne) + 0.015) / (1 - rhonce[h - 2])
    #                 continue
    #             if di == 1 and dj == h:
    #                 Dn[I,J] = RTT + ((Rs[J] * 8 / Ne) + 0.015) / (1 - rhonec[h - 2])
    #                 continue
    # return Dn, Tnce, Tnec

    