import numpy as np
from mV2mI import mV2mI

#   Create a service mesh call frequency matrix Fci

#   S : vector id Sid
#   M : number of microservices
#   e : number of datacenters
#   Fcm : microservice-level call frequency matrix
#   Fci : istance-level call frequency matrix 


def buildFci(S, Fcm, M, e):
    MN = M * e   # mesh nodes
    Fci = np.zeros((MN, MN))
    for i in range(M):
        for j in range(M):

            # cloud nodes
            if i != M-1:
                I = mV2mI(i, 1, M)
                J = mV2mI(j, 1, M)
                Fci[I, J] = Fcm[i, j]
            
            # edge nodes
            for h in range(2, e+1):
                I = mV2mI(i, h, M)
                J = mV2mI(j, h, M)
                if S[I] == 1:
                    if S[J] == 1:
                        Fci[I, J] = Fcm[i, j]
                    else:
                        Jc = mV2mI(j, 1, M)
                        Fci[I, Jc] = Fcm[i, j]
    return Fci

    # MN = M * e   # mesh nodes
    # Pci = np.zeros((MN, MN))
    # Ns = 2 ** MN

    # for i in range(M):
    #     for j in range(M):
    #         if i != M-1:
    #             # cloud
    #             I = mV2mI(i, 1, M)
    #             J = mV2mI(j, 1, M)
    #             Pci[I, J] = Pcm[i, j]
            
    #         for h in range(2, e+1):
    #             # edge nodes
    #             I = mV2mI(i, h, M)
    #             J = mV2mI(j, h, M)
    #             if S[I] == 1:
    #                 # microservice I running at edge h
    #                 if S[J] == 1:
    #                     # microservice J running at edge h
    #                     Pci[I, J] = Pcm[i, j]
    #                 else:
    #                     Jc = mV2mI(j, 1, M)    # cloud instance of j
    #                     Pci[I, Jc] = Pcm[i, j]
    
    # if dis:
    #     names = []
    #     color = []
    #     for I in range(MN):
    #         i, di = mI2mV(I, M)
    #         if di == 1:
    #             dis = "cloud"
    #             color.append([0.635294117647059, 0.0784313725490196, 0.184313725490196])
    #         else:
    #             dis = "edge"
    #             color.append([0.00, 0.45, 0.74])
    #         name = "<" + str(i) + "," + dis + ">"
    #         names.append(name)
        
    #     names[MN] = "user"
    #     G = nx.from_numpy_matrix(Pci)
    #     G = nx.relabel_nodes(G, dict(enumerate(names)))
        
    #     while True:
    #         to_remove = [node for node, degree in G.in_degree() if degree == 0]
    #         to_remove = [node for node in to_remove if node != "user"]  # user should not be removed
    #         G.remove_nodes_from(to_remove)
    #         color = [color[i] for i in range(len(color)) if i not in to_remove]
            
    #         if not to_remove:
    #             break
        
    #     #creategraphColor(G, color)

    # return Pci