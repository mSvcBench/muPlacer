import numpy as np

#   Create instance-level call frequency matrix Fci assuming 

#   S : vector id Sid, S[:M] (S[M:]) binary encoded presence vector for cloud (edge) microservices
#   M : number of microservices
#   Fcm : microservice-level call frequency matrix
#   Fci : instance-level call frequency matrix 


def buildFci(S, Fcm, M):
    MN = M * 2   # mesh nodes
    Fci = np.zeros((MN, MN))
    Fci[:M-1,:M-1] = Fcm[:M-1,:M-1]
    S_edge_id = np.argwhere(S[M:]==1).flatten()
    S_not_edge_id = np.argwhere(S[M:]==0).flatten()
    # temporary initialize all edge instances call cloud instances
    Fci[M:,:M] = Fcm[:,:]
    Fci[M+S_not_edge_id,:] = 0 # clean rows of edge instance not present @ edge
    y = np.repeat([S_edge_id], len(S_edge_id),axis=0).T
    Fci[M+S_edge_id,M+y] = Fcm[S_edge_id,y] # edge instances call edge insances
    Fci[M+S_edge_id,y] = 0 # clean edge cloud call for instances both at the edge 
    return Fci

    