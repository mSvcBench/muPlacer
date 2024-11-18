import numpy as np

#   Create instance-level call frequency matrix Fi assuming 

#   S : vector id Sid, S[:M] (S[M:]) binary encoded presence vector for cloud (edge) microservices
#   M : number of microservices
#   Fm : microservice-level call frequency matrix
#   Fi : instance-level call frequency matrix 


def buildFi(S, Fm, M):
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

    