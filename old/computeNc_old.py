import numpy as np
import time

#   Compute average number of time a microservice/instance is called per request

#   M : number of microservices
#   e : number of datacenters. e=1 compute Nc per microservice, e=2 compute Nc per instance
#   Fc : call frequency matrix 


def computeNcMat(Fc, M, e):
    # Compute Nc matrix through matrix inversion
    MN = len(Fc)
    H = -Fc.copy()
    np.fill_diagonal(H, 1)
    if e > 1:
        Ubit = np.arange(2, e+1) * M  # user position in the state vector
    else:
        Ubit = MN
    N = np.zeros(MN)
    N[Ubit-1] = 1
    H_inv = np.linalg.inv(H)
    Nc = np.dot(N, H_inv)
    Nc = np.array(Nc).flatten()
    return Nc

def computeNc(Fc, M, e):
    # Compute Nc vector through linear system solving
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

if __name__ == "__main__":
    
    M = 200
    # build dependency graph for testing
    Fcm = np.zeros([M,M])   # microservice call frequency matrix
    n_parents = 3
    for i in range(1,M-1):
        n_parent=np.random.randint(1,n_parents)
        for j in range(n_parent):
            a = np.random.randint(i)
            Fcm[a,i]=1
        
    # set random values for microservice call frequency matrix
    for i in range(0,M-1):
        for j in range(0,M-1):
            Fcm[i,j]=np.random.uniform(0.1,0.5) if Fcm[i,j]>0 else 0
    Fcm[M-1,0] = 1  # user call microservice 0 (the ingress microservice)
    
    for i in range(100):
        tic = time.time()
        NcMat = computeNc(Fcm, M,1)
        toc = time.time()
        print(f'processing time {toc-tic}')

        tic = time.time()
        NcLS = computeNc(Fcm, M,1)
        toc = time.time()
        print(f'processing time {toc-tic}')

        print(f'Equal computation {np.allclose(NcMat,NcLS)}')
        time.sleep(0.1)



